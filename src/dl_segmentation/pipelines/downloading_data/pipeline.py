from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=download_data,
                inputs=[],
                outputs="success",
                name="download_data_node",
            ),
        ]
    )
