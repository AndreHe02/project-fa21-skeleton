from parse import read_input_file, write_output_file, read_output_file
import os
import numpy as np # for fast exponentiation
import random

decay = np.array(list(range(1440)))
decay = np.exp(decay * -0.0170)

def ensemble_solve(tasks):
    Ws = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
    best = None
    best_r = 0
    best_idx = 0
    for i, w in enumerate(Ws):
        t = list(tasks)
        out = solve(t, w)
        sc = evaluate(tasks, out)
        if sc > best_r:
            best_r = sc
            best = out
            best_idx = i
    return best

def simple_solve(tasks):

    orders = [sorted(tasks, key=lambda t: t.deadline),
        sorted(tasks, key=lambda t: -t.deadline),
        sorted(tasks, key=lambda t: -t.perfect_benefit),
        sorted(tasks, key=lambda t: t.perfect_benefit)]

    order = list(tasks)
    random.shuffle(order)
    orders.append(order)

    best_r = 0
    best = None
    for i, o in enumerate(orders):
        out = solve(tasks, 0, o)
        sc = evaluate(tasks, out)
        if sc > best_r:
            best_r = sc
            best = out

    return best


def solve(tasks, W, task_ls=None):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing
    """

    def sortkey(t):
        #r = t.perfect_benefit
        #ex = -np.log(r / 50) / 0.0170
        #return t.deadline + ex
        return t.deadline - W * t.perfect_benefit

    if not task_ls:
        tasks = sorted(tasks, key=sortkey)
    else:
        tasks = task_ls

    dp = np.zeros((len(tasks), 1440))
    prev = np.zeros((len(tasks), 1440, 2))
    for i in range(len(tasks)):
        for t in range(1440):
            task = tasks[i]
            do = 0
            dont = 0
            if i > 0:
                dont = dp[i-1, t]
            if task.duration <= t:
                if t > task.deadline:
                    do = dp[i-1, t-task.duration] + task.perfect_benefit * decay[t-task.deadline]
                else:
                    do = dp[i-1, t-task.duration] + task.perfect_benefit

            dp[i, t] = max(do, dont)
            if do > dont:
                prev[i, t, 0] = i-1
                prev[i, t, 1] = t - task.duration
            else:
                prev[i, t, 0] = i-1
                prev[i, t, 1] = t
    mx_rw = 0
    mx_t = 0
    for t in range(1440):
        if dp[-1, t] > mx_rw:
            mx_rw = dp[-1, t]
            mx_t = t

    order = []
    i, t = len(tasks)-1, mx_t
    while i != 0 or t != 0:
        pi, pt = int(prev[i, t, 0]), int(prev[i, t, 1])
        if pt < t:
            order.append(tasks[i].task_id)
        i, t = pi, pt
    order.reverse()

    while True:
        mx_last = 0
        last = None
        for task in tasks:
            if task.task_id not in order:
                if task.duration <= 1440 - mx_t:
                    if task.deadline >= mx_t + task.duration:
                        rew = task.perfect_benefit
                    else:
                        rew = task.perfect_benefit * decay[mx_t + task.duration-task.deadline]
                    if rew > mx_last:
                        mx_last = rew
                        last = task
        if last is not None:
            order.append(last.task_id)
            mx_t += last.duration
        else:
            break
    return order

def baseline(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing
    """

    tasks = sorted(tasks, key=lambda t: t.deadline)

    dp = np.zeros((len(tasks), 1440))
    prev = np.zeros((len(tasks), 1440, 2))
    for i in range(len(tasks)):
        for t in range(1440):
            task = tasks[i]
            do = 0
            dont = 0
            if i > 0:
                dont = dp[i-1, t]
            if task.duration <= t:
                if t > task.deadline:
                    do = dp[i-1, t-task.duration] + task.perfect_benefit * decay[t-task.deadline]
                else:
                    do = dp[i-1, t-task.duration] + task.perfect_benefit

            dp[i, t] = max(do, dont)
            if do > dont:
                prev[i, t, 0] = i-1
                prev[i, t, 1] = t - task.duration
            else:
                prev[i, t, 0] = i-1
                prev[i, t, 1] = t
    mx_rw = 0
    mx_t = 0
    for t in range(1440):
        if dp[-1, t] > mx_rw:
            mx_rw = dp[-1, t]
            mx_t = t

    order = []
    i, t = len(tasks)-1, mx_t
    while i != 0 or t != 0:
        pi, pt = int(prev[i, t, 0]), int(prev[i, t, 1])
        if pt < t:
            order.append(tasks[i].task_id)
        i, t = pi, pt
    order.reverse()
    return order

def evaluate(tasks, output):
    max_task_id = max([task.task_id for task in tasks])
    idxs = [None for _ in range(max_task_id+1)]
    for task in tasks:
        idxs[task.task_id] = task
    score = 0
    time = 0
    for id in output:
        task = idxs[id]
        time += task.duration
        if time > 1440:
            break
        if time > task.deadline:
            score += task.perfect_benefit * decay[time - task.deadline]
        else:
            score += task.perfect_benefit
    return score

'''
if __name__ == '__main__':
    num_files = 0
    total_score = 0
    for input_path in os.listdir('inputs/small/')[:50]:
        if not input_path.endswith('.in'):
            continue
        tasks = read_input_file('inputs/small/'+input_path)
        output = ensemble_solve(tasks)
        score = evaluate(tasks, output)
        b = baseline(tasks)
        bscore = evaluate(tasks, b)
        print(score, bscore)
        total_score += score - bscore
        num_files += 1
    print(count)
    print(total_score / num_files)
'''
# Here's an example of how to run your solver.

from tqdm import tqdm
if __name__ == '__main__':
    for size in ['small', 'medium', 'large']:
         for input_path in tqdm(os.listdir('inputs/%s/' %size )):
             if not input_path.endswith('.in'):
                 continue
             output_path = 'outputs/%s/' % size + input_path[:-3] + '.out'
             tasks = read_input_file('inputs/%s/' % size + input_path)
             old_output = read_output_file(output_path)
             old_score = evaluate(tasks, old_output)

             output = ensemble_solve(tasks)
             #output = simple_solve(tasks)
             score = evaluate(tasks, output)
             if score > old_score:
                 print('improved')
                 write_output_file(output_path, output)
