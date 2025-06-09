import logging

def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    # if hasattr(logging, levelName):
    #    raise AttributeError('{} already defined in logging module'.format(levelName))
    # if hasattr(logging, methodName):
    #    raise AttributeError('{} already defined in logging module'.format(methodName))
    # if hasattr(logging.getLoggerClass(), methodName):
    #    raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

def log_wrapper(logger, message, *args, exception_info=False, level='usky_warn'):
    if level.lower() == 'critical': logger.critical(message, *args, exc_info=exception_info)        # level 50
    if level.lower() == 'error': logger.error(message, *args, exc_info=exception_info)              # level 40

    if level.lower() == 'usky_warn': logger.usky_warn(message, *args, exc_info=exception_info)      # level 37
    if level.lower() == 'usky_info': logger.usky_info(message, *args, exc_info=exception_info)      # level 35
    
    if level.lower() == 'warning': logger.warning(message, *args, exc_info=exception_info)          # level 30

    if level.lower() == 'usky_debug': logger.usky_debug(message, *args, exc_info=exception_info)    # level 25

    if level.lower() == 'info': logger.info(message, *args, exc_info=exception_info)                # level 20
    if level.lower() == 'debug': logger.debug(message, *args, exc_info=exception_info)              # level 10

def parprint(*args,**kwargs):
    import sys
    print("".join(map(str,args)),**kwargs);  sys.stdout.flush()

def profiletime(task_tag, step, times, comm=None, mpiproc=0):
    """
    Profile and print timing for a specific step.
    
    Parameters:
    -----------
    task_tag : str or None
        Task identifier for custom messaging
    step : str
        Name of the step being timed
    times : dict
        Dictionary to store timing data
    comm : MPI communicator or None
        MPI communicator for synchronization
    mpiproc : int
        MPI process rank (only rank 0 prints output)
        
    Returns:
    --------
    times : dict
        Updated timing dictionary
    """
    from time import time
    if comm is not None:
        comm.Barrier()

    stepn = step + '_N'
    dt = time() - times['t0']
    
    if step in times.keys():
        times[step] += dt
        times[stepn] += 1
    else:
        times[step] = dt
        times[stepn] = 1
    
    times['t0'] = time()

    if mpiproc != 0:
        return times

    # Format step output with consistent styling
    if task_tag is not None:
        parprint(f'⏱️  {task_tag}: {dt:.3f}s (iteration {times[stepn]}, {step})')
    else:
        # Format step name for better readability
        formatted_step = step.replace('_', ' ').replace('computation', '').strip().title()
        parprint(f'⏱️  {formatted_step}: {dt:.3f}s')

    return times

def _sortdict(dictin,reverse=False):
    return dict(sorted(dictin.items(), key=lambda item: item[1], reverse=reverse))

def summarizetime(task_tag, times, comm=None, mpiproc=0):
    """
    Print a formatted summary of timing information.
    
    Parameters:
    -----------
    task_tag : str or None
        Task identifier (not used in current implementation)
    times : dict
        Dictionary containing timing data with keys for each step and '_N' suffix for counts
    comm : MPI communicator or None
        MPI communicator for synchronization
    mpiproc : int
        MPI process rank (only rank 0 prints output)
    """
    total_time = 0

    if comm is not None:
        comm.Barrier()
    if mpiproc != 0:
        return times

    # Calculate total time and format output
    step_times = []
    for key in times.keys():
        if key != 't0' and key[-2:] != '_N':
            if key + '_N' in times:
                N = times[key + '_N']
                step_time = times[key]
                total_time += step_time
                
                # Format step name for better readability
                formatted_key = key.replace('_', ' ').title()
                step_times.append((formatted_key, step_time, N))

    # Sort by time (longest first)
    step_times.sort(key=lambda x: x[1], reverse=True)

    # Print formatted timing summary
    print(f"\n📊 Timing Summary:")
    print(f"   {'Step':<25} {'Time (s)':<10} {'Avg/Iter':<12}")
    print(f"   {'-'*25} {'-'*10} {'-'*12}")
    
    for step_name, step_time, iterations in step_times:
        avg_time = step_time / iterations if iterations > 0 else step_time
        print(f"   {step_name:<25} {step_time:>8.3f}s   {avg_time:>8.3f}s")
    
    print(f"   {'-'*25} {'-'*10} {'-'*12}")
    print(f"   {'Total Runtime':<25} {total_time:>8.3f}s")
    print()