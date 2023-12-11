import torch
from thop import profile
import numpy as np


def getGFLOPs(model, input):
    flops, params = profile(model, inputs=(input,))
    FLOPs = str(flops/1e9) # 单位是G
    params = str(params/1e6) # 单位是M
    return FLOPs, params

def getInferenceS(model, input):
    dummy_input = input
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000 # 5000
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(100):
       _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
      for rep in range(repetitions):
         starter.record()
         _ = model(dummy_input)
         ender.record()
         # WAIT FOR GPU SYNC
         torch.cuda.synchronize()
         curr_time = starter.elapsed_time(ender)
         timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    return mean_syn, mean_fps