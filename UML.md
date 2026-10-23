classDiagram
    class LinearRegression {
        +total_model_count : int $
        -_initializeParam()
        +fit(epochs: int)
        +predict(feature: int)  float
        -_loss() float
    }
    
    Model <|-- LinearRegression
    LinearRegression ..> Optimizer : uses
