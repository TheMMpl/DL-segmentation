from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_model,
    prepare_dataset,
    run_inference,
    save_results,
)


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return pipeline(
        [
            node(
                func=load_model,
                inputs="params:model_checkpoint",
                outputs="model",
                name="load_model_node",
            ),
            node(
                func=prepare_dataset,
                inputs="params:image_amount",
                outputs=["val_snippet","image_snippet"],
                name="load_data_node",
            ),
            node(
                func=run_inference,
                inputs=["model","val_snippet"],
                outputs=["result","IoU"],
                name="run_inference_node",
            ),
            node(
                func=save_results,
                inputs=["result","image_snippet","IoU"],
                outputs=["results","IoU_metrics"],
                name="save_results_node",
            ),
        ]
    )
