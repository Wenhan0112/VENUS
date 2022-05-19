from typing import Dict, List
import skopt
import numpy as np
from typing import Callable
import time

try:
    import venus_utils.venusplc as venusplc
    venus = venusplc.VENUSController('/home/damon/venus_data_project/config/config.ini')
except:
    print("VENUS PLC controller is not loaded. ")
    pass


def change_superconductors(Igoal):
    time_start_change = time.time()
    usefastdiff = 0.1  # only use the fast search if the difference between current and goal is > this amount
    Inow=np.array([venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i'])])   # current currents

    done = np.zeros(3)+1
    direction = np.sign(Igoal-Inow)
    Idiff = np.abs(Igoal-Inow)
    done[np.where(Idiff>usefastdiff)]=0

    Iaim = np.zeros(3)
    Iaim[np.where(direction>0)]=Igoal[np.where(direction>0)]+5
    Iaim[np.where(direction<0)]=Igoal[np.where(direction<0)]-5
    Iaim[np.where(done==1)]=Igoal[np.where(done==1)]

    diffup = np.array([.03,.04,.08])
    diffdown = np.array([.06,.10,.25])
    Ioff = Igoal*1.0
    for i in range(len(Ioff)):
        if direction[i]>0: Ioff[i]=Ioff[i]-direction[i]*diffup[i]
        if direction[i]<0: Ioff[i]=Ioff[i]-direction[i]*diffdown[i]

    print('\nin change:\nInow=',Inow)
    print('Iaim=',Iaim)
    print('Ioff=',Ioff)

    checkdone = np.zeros((3,40))+5.0

    start_time = time.time()
    venus.write({'inj_i':Iaim[0], 'ext_i':Iaim[1], 'mid_i':Iaim[2] })
    #print('starting new field setting')

    def check_done(done,Inow,Igoal,Ioff):
        if done[0]==0 and direction[0]*(Inow[0]-Ioff[0])>0:
            venus.write({'inj_i':Igoal[0]}); done[0]=1
            print('inj to goal:',done,' Inow:',Inow[0],' Igoal:',Igoal[0] )
        if done[1]==0 and direction[1]*(Inow[1]-Ioff[1])>0:
            venus.write({'ext_i':Igoal[1]}); done[1]=1
            print('ext to goal:',done,' Inow:',Inow[1],' Igoal:',Igoal[1] )
        if done[2]==0 and direction[2]*(Inow[2]-Ioff[2])>0:
            venus.write({'mid_i':Igoal[2]}); done[2]=1
            print('mid to goal:',done,' Inow:',Inow[2],' Igoal:',Igoal[2] )
        return(done)

    names=['inj_i','ext_i','mid_i']
    diffall = len(checkdone[0,:])*.04
    while np.sum(checkdone[0,:])>diffall or np.sum(checkdone[1,:])>diffall or np.sum(checkdone[2,:])>diffall:
        for i in range(5):
            time.sleep(0.1)
            Inow=np.array([venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i'])])   # current currents
            done = check_done(done,Inow,Igoal,Ioff)

        if time.time()-time_start_change >300.0:
            print('!!!!!!!!!  timed out !!!!!!!')
            print('trying to set',Igoal)
            print('got stuck at ',venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i']))   # current currents
            Inow=np.array([venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i'])])   # current currents
            for i in range(3):
                print('in here. i=%i, Inow[i]=%.2f, Igoal[i]=%.2f, done[i]=%i'%(i,Inow[i],Igoal[i],done[i]))
                if np.abs(Inow[i]-Igoal[i])<0.08 and done[i]==0:   # for small change problem
                    Igoal[i]=Igoal[i]-.01*np.sign(Inow[i]-Igoal[i])
                    venus.write({names[i]:Igoal[i]})
                    print('stuck with small difference.  resetting Igoal')
                    time_start_change=time.time()
                if np.abs(Inow[i]-Igoal[i])>=0.08:
                    done[i]=0
                    venus.write({names[i]:Igoal[i]})
                    print('stuck with big difference. re-requesting Igoal')
                    time_start_change=time.time()
        checkdone[:,:-1] = checkdone[:,1:]; checkdone[:,-1]=np.abs(Inow-Igoal)
    #print('done setting field ')



def monitor(t_start, output_file, program_start_time, delta_time=60.):
    print('monitoring...')
    loop_time = time.time()
    while time.time() < t_start + delta_time:

        if 0:
            Ifc = venus.read(['fcv1_i'])*1e6        # faraday cup current (single species current) in microamps
            Idrain = venus.read(['extraction_i'])   # drain current ~ total extracted beam current [mA]
            Pinj = venus.read(['inj_mbar'])         # injection pressure [torr]

            # things also worth monitoring to understand system
            Ibias = venus.read(['bias_i'])          # bias disk current [mA]
            Ipull = venus.read(['puller_i'])        # puller electrode current [mA]
            Xsrc = venus.read(['x_ray_source'])     # amount of x-rays produced by source [?]
            Pext = venus.read(['ext_mbar'])         # pressure just outside source [torr]
            HHe = venus.read(['four_k_heater_power'])   # liquid He heater power [W]

        # TODO: write to database or somewhere
        output_file.write("%7.1f %12.3f %12.3f %10.4f %10.4f %8.2e %8.2e %7.2f %7.2f %7.2f %7.2f %7.1f\n"%(time.time()-loop_time,
            time.time()-program_start_time,venus.read(['fcv1_i'])*1e6,venus.read(['extraction_i']),venus.read(['bias_i']),
            venus.read(['inj_mbar']),venus.read(['ext_mbar']),venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i']),
            venus.read(['sext_i']),venus.read(['x_ray_source'])))
        time.sleep(1)


def plc_control(input: Dict[str, float], output_names, output_file, program_start_time, delta_time) -> \
    Dict[str, float]:
    """
    The objective function that is to be maximized.
    @params input (Dict[str, float]): The input name-value pairs.
        input[a]: The value of item A that is to be input to the VENUS PLC.
    @params output_names: The output item names.
    @params output_file (str): A file to write monitoring values too.
    @params delta_time (float): The duration we want to monitor for in seconds.
    @return (Dict[str, float]): The output name-value pairs.
        RETURN[a]: The value of item A that is output by the VENUS PLC.
        Constraint: list(RETURN.keys()) == output_names
    """
    # This is to be changed for faster implementation, i.e, overshoot.
    # dst: replacing with below  venus.write(input) # May use setpoints for current control?
    Igoal = np.array([input[k] for k in ['inj_i', 'ext_i', 'mid_i']])
    t0 = time.time()
    change_superconductors(Igoal)
    t1 = time.time()
    print(t1-t0, ' seconds to set superconductors')

    monitor(t1, output_file, program_start_time, delta_time)
    t2 = time.time()
    print(t2-t1, ' seconds to set superconductors')
    assert len(output_names) == 1

    t_end = time.time() + 10 # data acquisition for 10s
    v_list = []
    while time.time() < t_end:
        v = venus.read(output_names)
        time.sleep(0.25)
        v_list.append(v)
    v_mean = sum(v_list) / len(v_list)
    t3 = time.time()
    print(t3-t2, ' seconds to do averaging')
    # save the v list?
    print('average current for 10 s: ',v_mean)
    return {'fcv1_i': v_mean}

def input_transform(input_names: List[str]) -> \
    Callable[[List[float]], Dict[str, float]]:
    """
    Transform the input to a venus identifiable dictionary.
    @params input_names (List[str]): A list of items that are the input to the
        VENUS PLC.
    @return (Callable[[List[float]], Dict[str, float]]): A function matches the
        item name with the value.
        Input (List[float]): A list of values that is in sequence with the
            INPUT_NAMES.
        Output (Dict[str, float]): A dictionary where the item name and value
            are matched.
    """
    def transform_fn(input_vals: List[float]) -> Dict[str, float]:
        """
        Match the item value with the name as a dictionary.
        @params (List[float]): A list of values that is insequence with the
            INPUT_NAMES.
        @return (Dict[str, float]): A dictionary where the item name and value
            are matched.
        """
        return dict(zip(input_names, input_vals))

    return transform_fn


def objective_func(output: Dict[str, float]) -> float:
    """
    Transform the output to a loss function.
    @params output (Dict[str, float]): The output dictionary.
        output[a]: The measurement of item A from the VENUS PLC.
    @return (float): The function to be maximized.
    """
    return -output['fcv1_i'] * 1e6


def main():
    """
    The main function.
    """

    program_start_time = time.time()
    name_start_time = str(int(program_start_time))

    """
    num_calls (int): Number of sampling that occurs
    Constraint: num_calls > 0
    """
    num_calls = 30

    """
    x0 (Optinal[List[List[float]]]): Initial sampling sites
    It is directly fed into skopt.gp_minimize
    """
    x0 = None

    """
    num_init (int): Initial samples
    Constraint: num_init >= 0
    Constraint: num_init < num_calls - len(x0) if x0 is not None else num_calls
    """
    num_init = 10

    """
    input_names (List[str]): Input item names to the VENUS PLC.
    They should be specified in the configuration file of the VENUS PLC.
    """
    input_names = ['inj_i', 'ext_i', 'mid_i']

    """
    input_ranges (Dict[str, tuple(int)]): Input ranges
    They should be compatible as in the configuration file of the VENUS PLC.
    Constraint: all(l < h for l, h in input_ranges.values())
    """
    input_ranges = {
        "inj_i": (120, 130),
        "ext_i": ( 97, 110),
        "mid_i": ( 95, 107)
    }

    """
    output_names (List[str]): Request output item name of the VENUS PLC.
    They should be specified in the configuration file of the VENUS PLC.
    """
    output_names = ["fcv1_i"]

    """
    is_test (bool): True iff use test function.
    """
    is_test = False

    # Transform function from the objective function input to the PLC
    # Controller input.
    transform_fn = input_transform(input_names)

    # Dimension fed into skopt.gp_minimize
    dimension = [
        skopt.space.space.Real(*input_ranges[key])
        for key in input_names
    ]

    with open('monitor_wenhan_'+name_start_time,'w') as writefile:

        # The function fed to skopt.gp_minimize for optimization.
        def opt_function(input):
            return objective_func(
                plc_control(transform_fn(input), output_names, writefile, program_start_time, 1. * 60.)
            )

        if is_test:
            """
            Test function on x: Himmelblau's function
            Range: $[-5, 5] \times [-5, 5]$
            """
            def opt_function(input):
                x, y = input[0], input[1]
                return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
            dimension = [skopt.space.space.Real(-5, 5)] * 2

        # Minimizing model.
        print("Minimizing objective function. ")
        model = skopt.gp_minimize(
            opt_function,
            dimension,
            n_calls = num_calls,
            x0 = x0,
            n_initial_points = num_init
        )

        print("Minimum location:", model.x)

if __name__ == "__main__":
    main()
