from pipeline import run_pipeline

def test_pipeline_accuracy_threshold():
    accuracy = run_pipeline('data/iris.csv')
    assert accuracy >=0.85
