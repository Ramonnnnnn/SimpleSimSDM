import numpy as np


# def generate_poisson_events(load_erlangs, average_holding_time, attempt_n_calls):
#     # Using offered load in E to calculate
#     call_arrive_rate = load_erlangs / average_holding_time
#     # Total (simulation) time duration will depend on the arrival rate and number of desired calls to be attempted
#     time_duration = attempt_n_calls / call_arrive_rate
#     # Calculated so that it is = to attempt_n_calls, the number of calls I want to exist in my simulation
#     num_events = np.random.poisson(call_arrive_rate * time_duration)
#     event_times = np.sort(np.random.uniform(0, time_duration, num_events))
#     inter_arrival_times = np.diff(event_times)
#     return num_events, event_times, inter_arrival_times, time_duration
#
#
# num_events, event_times, inter_arrival_times, time_duration = generate_poisson_events(50, 1, 10000)
#
# print(f"Number of events: {num_events}, Event_times: {len(event_times)}, {event_times[1]}, inter arrival times {inter_arrival_times[0]}. time duration: {time_duration}")
#
