from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=[],
                outputs=["unet","trainer"],
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["unet","trainer"],
                name="evaluate_model_node",
                outputs="metrics",
            ),
        ]
    )
