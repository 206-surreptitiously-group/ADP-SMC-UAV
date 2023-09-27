# ADP-SMC-UAV
A repository about ADP-based SMC controller for a UAV.
 
To satisfy the requirements of physical experiments. We use SMC to design both inner and outer-loop controllers.
However, we fix the inner-loop controller parameters and just use RL to train some hyperpmeters of the outer loop controller.

Therefore, the uav model, inner-loop controller, and outer-loop controller are integrated together as the "environment" of the RL.

Hope we can get satisfactory performance as soon as possible.
