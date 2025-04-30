import mindspore
from typing import Union, cast

from vllm.multimodal.inputs import BatchedTensorInputs, JSONTree, json_map_leaves


NestedTensors = Union[list["NestedTensors"], list[mindspore.Tensor], mindspore.Tensor,
                      tuple[mindspore.Tensor, ...]]


@staticmethod
def as_kwargs(
    batched_inputs: BatchedTensorInputs,
    *,
    device = None,
) -> BatchedTensorInputs:
    json_inputs = cast(JSONTree[mindspore.Tensor], batched_inputs)

    json_mapped = json_map_leaves(
        lambda x: x,
        json_inputs,
    )

    return cast(BatchedTensorInputs, json_mapped)
