from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    check_model_inference,
    create_demo_dir,
)


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return pipeline(
        [
            node(
                func=create_demo_dir,
            ),
            node(
                func=check_model_inference,
            ),
        ]
    )
