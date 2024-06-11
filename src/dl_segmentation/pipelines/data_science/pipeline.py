from kedro.pipeline import Pipeline, node, pipeline

from .nodes import test_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=[],
                outputs=["unet","trainer"],
                name="train_model_node",
            ),
            node(
                func=test_model,
                inputs=["unet","trainer"],
                name="test_model_node",
                outputs="metrics",
            ),
        ]
    )
