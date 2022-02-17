import torch
import warnings
from typing import Any, Iterable, List, Tuple
from torch.autograd.variable import Variable
from uu.utils import memory 


def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")

def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states) -> None:
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)

def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)



class cCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        #print("in customized checkpoint forward", len(args))
        #print("args", args)
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)
        # TODO: assume first arg always be input tensor
        # the very begining tile will be accumulatted !! have to create a fork node
        # fork node ???
        
        # how about multiple inputs 
        # lets restrict last one is info
        # args[0].requires_grad=True      #??
        #print("input size",args[0].size() )
        ctx.save_for_backward(args[0])
        ctx.payload = args[1:]
        is_ccheckpoint = True
        args = list(args)
        args.append(is_ccheckpoint)
        args = tuple(args)
        
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs
    
    @staticmethod
    def backward(ctx, *args):
        # 1) get the tile from cpu
        # 2) fwd per tile
        # 3) bwd 
        #print("\n############# Enter checkpointing bkward ####")
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.saved_tensors
        payload = list(ctx.payload)
        inputs = list(inputs)
        inputs.extend(payload)
        #print("inputs len", len(inputs), len(payload))


        inputs = tuple(inputs)
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # memUsage = memory.MeasureMemory(device)
        # print("==== before bkloop ...")
        # initmem = memUsage.currentValue()
        # print(memory.MemSize(initmem))      #now should be around 3.8MB
        
        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(inputs)
            
            with torch.enable_grad():
                #print("ctx.run_function bkw", ctx.run_function)
                outputs = ctx.run_function(*detached_inputs)


        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # memUsage = memory.MeasureMemory(device)
        # print("==== bkloop ...")
        # initmem = memUsage.currentValue()
        # print(memory.MemSize(initmem))      

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # print("#############", len(outputs))
        # print(args)
        # print (outputs[0].size())
        # print (args[0].size())
        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary")

        torch.autograd.backward(outputs_with_grad, args_with_grad)

        # eh.. how to pass this info to here
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in detached_inputs)
        #print("HREREERE", grads[0].size())
        
        res = (None, None) + grads
        return  res


def checkpoint(function, *args, **kwargs):
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    return cCheckpoint.apply(function, preserve, *args)