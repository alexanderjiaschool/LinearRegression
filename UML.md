classDiagram
    class LinearRegression {
        +total_model_count : int$
        +features : ndarray
        +y : ndarray
        #_param : dict
        -_optimizer : Optimizer
        +__init__(features: list, y: list, optimizer: Optimizer)
        -_increaseModelCount()$
        +getModelCount()$ int
        +standardTemplate()$ LinearRegression
        -_initializeParam()
        +fit(epochs: int)
        +predict(features) ndarray
        -_loss() float
        +mean_squared_error(y, y_hat)$ float
        #_gradients() dict
        -_isValidShape(features, y) bool
    }

    class LassoRegression {
        -_a : float
        +__init__(features: list, y: list, optimizer: Optimizer, alpha: float)
        #_gradients() dict
    }

    class RidgeRegression {
        -_a : float
        +__init__(features: list, y: list, optimizer: Optimizer, alpha: float)
        #_gradients() dict
    }

    class Optimizer {
        #_learning_rate : float
        #_step_count : int
        #_n : int
        +__init__(learning_rate: float, n: int)
        +checkShape(param: dict, gradients: dict)
        +step(param: dict, gradients: dict)
    }

    class Adam {
        +beta1 : float
        +beta2 : float
        -_m_w : ndarray
        -_v_w : ndarray
        -_m_b : float
        -_v_b : float
        +m_w_corrected : ndarray
        +v_w_corrected : ndarray
        +m_b_corrected : float
        +v_b_corrected : float
        +__init__(learning_rate: float, n: int, beta1: float, beta2: float)
        +correction(x, beta, exp)$
        +step(param: dict, gradients: dict)
    }

    class Momentum {
        -_beta : float
        -_v : dict
        +__init__(learning_rate: float, n: int, beta: float)
        +step(param: dict, gradients: dict)
    }

    LinearRegression <|-- LassoRegression
    LinearRegression <|-- RidgeRegression
    Optimizer <|-- Adam
    Optimizer <|-- Momentum
    LinearRegression ..> Optimizer : uses
