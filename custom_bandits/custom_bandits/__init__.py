from gym.envs.registration import register

register(id='twoArm_bandits-v0', entry_point='custom_bandits.envs:TwoArmBandit')
register(id='tenArmGaussian_bandits-v0', entry_point='custom_bandits.envs:TenArmGaussianBandit')