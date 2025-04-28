import os
from typing import Dict, Any, List, Tuple
import numpy as np
import tensorrt as trt
import cuda  # from cuda-python pkg
import pycuda.driver as cuda_drv
import pycuda.autoinit           # одноразова ініціалізація CUDA-контексту

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def _dtype_of(tensor_dtype: trt.DataType):
    """TensorRT dtype → NumPy dtype."""
    return {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int32:   np.int32,
        trt.int8:    np.int8,
        trt.bool:    np.bool_,
    }[tensor_dtype]


class TrtRunner:
    """
    Загальний раннер для будь-якого TensorRT-двигуна (.trt).

    ● Підтримує dynamic shape та декілька входів/виходів.
    ● Розподіляє pinned-host та device-пам’ять, кешує її за формою.
    ● API:
        >>> runner = TrtRunner("model.trt")
        >>> outputs = runner(inputs_dict)          # dict name→np.ndarray
    """

    def __init__(self, engine_path: str, device_id: int = 0):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        self.stream = cuda_drv.Stream()
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(f.read())
        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        self.device_id = device_id

        # I/O-метадані
        self.input_names: List[str] = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        ]
        self.output_names: List[str] = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.OUTPUT
        ]

        # Кеш буферів keyed by shape tuple
        self._buffer_cache: Dict[Tuple[Tuple[int, ...], ...], Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _allocate_buffers(self, input_shapes: Dict[str, Tuple[int, ...]]):
        """
        Внутрішнє: виділяє pinned-host та device-пам’ять під конкретну
        комбінацію форм. Результат кешується, щоб не робити malloc на кожен батч.
        """
        cache_key = tuple(sorted(input_shapes.items()))
        if cache_key in self._buffer_cache:
            return self._buffer_cache[cache_key]

        bindings: Dict[str, Dict[str, Any]] = {}
        for name in self.engine:
            mode = self.engine.get_tensor_mode(name)
            dtype = _dtype_of(self.engine.get_tensor_dtype(name))
            shape = (
                input_shapes[name]
                if mode == trt.TensorIOMode.INPUT
                else self.engine.get_tensor_shape(name)
            )

            # Для output із dynamic dim залежних від input батчу ―
            # форму уточнимо вже після set_input_shape + resolve.
            vol = int(np.prod(shape)) if -1 not in shape else 0
            host_mem = (
                cuda_drv.pagelocked_empty(vol, dtype) if vol > 0 else None
            )  # виділимо пізніше
            dev_mem = (
                cuda_drv.mem_alloc(host_mem.nbytes) if host_mem is not None else None
            )

            bindings[name] = {
                "host": host_mem,
                "device": dev_mem,
                "dtype": dtype,
                "shape": shape,
                "mode": mode,
            }

        # збережемо
        self._buffer_cache[cache_key] = bindings
        return bindings

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __call__(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Запуск інференсу.

        Args:
            inputs: dict {tensor_name: np.ndarray} ― форми/типи мають відповідати engine.

        Returns:
            dict {output_name: np.ndarray}
        """
        # ---- 1. прописуємо dynamic shapes ----
        for name, arr in inputs.items():
            self.context.set_input_shape(name, tuple(arr.shape))
        assert self.context.all_binding_shapes_specified

        # ---- 2. готуємо (або відновлюємо з кешу) буфери ----
        input_shapes = {n: tuple(arr.shape) for n, arr in inputs.items()}
        bindings = self._allocate_buffers(input_shapes)

        # ---- 3. встановлюємо адреси тензорів ----
        for name, buf in bindings.items():
            if buf["device"] is None:  # output, форму тепер знаємо
                shape = tuple(self.context.get_tensor_shape(name))
                vol = int(np.prod(shape))
                host_mem = cuda_drv.pagelocked_empty(vol, buf["dtype"])
                dev_mem = cuda_drv.mem_alloc(host_mem.nbytes)
                buf["host"], buf["device"], buf["shape"] = host_mem, dev_mem, shape
            self.context.set_tensor_address(name, int(buf["device"]))

        # ---- 4. копіюємо входи host→device ----
        for name, arr in inputs.items():
            buf = bindings[name]
            np.copyto(buf["host"].reshape(arr.shape), arr)
            cuda_drv.memcpy_htod_async(buf["device"], buf["host"], self.stream)

        # ---- 5. inference ----
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # ---- 6. device→host для виходів ----
        outputs: Dict[str, np.ndarray] = {}
        for name in self.output_names:
            buf = bindings[name]
            cuda_drv.memcpy_dtoh_async(buf["host"], buf["device"], self.stream)
        self.stream.synchronize()

        for name in self.output_names:
            buf = bindings[name]
            outputs[name] = buf["host"].reshape(buf["shape"]).copy()  # detach

        return outputs
