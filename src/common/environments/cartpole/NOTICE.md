# CartPole attribution

Copied from Gymnasium 1.4.0, `gymnasium/envs/classic_control/cartpole.py`.
Upstream: https://github.com/Farama-Foundation/Gymnasium
The upstream MIT license is included in LICENSE.

This copy retains the scalar environment and renderer. Changes add validated
constructor parameters for cart mass, pole mass, and viscous hinge damping.
The upstream vector environment and its documentation are omitted.

Let l be the half-length and M the total mass. The coupled equations are:

- M x_acc + m l cos(theta) theta_acc = F + m l sin(theta) theta_velocity^2
- (4/3) m l^2 theta_acc + m l cos(theta) x_acc = m g l sin(theta) - b theta_velocity

Solving these equations adds torque / (m l) to the standard angular-acceleration
numerator. Cart acceleration then uses that angular acceleration. Zero damping
recovers the upstream equations. The installed Gymnasium package is untouched.
