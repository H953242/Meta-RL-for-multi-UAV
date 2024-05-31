from runner import Runner
from common.arguments import get_args
from final_env_100u_meta import Environ
import time


if __name__ == '__main__':
    # get the params
    # print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    args = get_args()
    env = Environ()
    env.seed(2020)
    env.policy = "drl"
    # 补充参数
    args.n_agent = env.n_agent
    args.n_actions = env.action_range
    args.state_dim = env.state_dim
    args.action_dim = env.action_dim
    # 运行
    runner = Runner(args, env)
    runner.run()
    runner.plot()

