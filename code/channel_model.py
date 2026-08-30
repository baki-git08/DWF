import numpy as np
from DWF_v2 import power

thz_freq = np.array([0.30, 0.45, 0.48, 0.60, 0.66, 0.90, 0.99, 1.05, 
                     1.26, 1.29, 1.31, 1.40, 1.53, 1.56, 1.83, 1.98, 
                     2.13, 2.52, 2.84, 3.09, 3.42, 3.72])

thz_alpha_cm = np.array([123, 148, 156, 167, 180, 210, 220, 237, 256, 
                         269, 271, 275, 294, 295, 326, 347, 367, 433, 
                         502, 549, 622, 739])

thz_alpha_m = thz_alpha_cm * 100 

distance_m = 0.001  # Example distance in meters


class ChannelModel:
    def __init__(self, distance_m=distance_m): 
        self.distance_m = distance_m
        # self.electromagnetic_wave_types = ['THz', 'Optics']  # Supported wave types

    def beer_lambert_attenuation(self, alpha):
        '''Total attenuation over self.distance_m for a given absorption coefficient alpha.'''
        return np.exp(-alpha * self.distance_m)

    def get_thz_attenuation(self, frequency):
        if frequency < thz_freq[0] or frequency > thz_freq[-1]:
            raise ValueError("Frequency is out of the range of the model.")
        # Interpolate to find the attenuation coefficient for the given frequency
        alpha = np.interp(frequency, thz_freq, thz_alpha_m)
        print(f"Interpolated THz attenuation coefficient (alpha) for frequency {frequency} THz: {alpha} m^-1")
        return self.beer_lambert_attenuation(alpha)

    def get_optics_attenuation(self, frequency):
        # Placeholder for optics attenuation model
        # Implement the optics attenuation coefficient lookup here, then:
        # return self.beer_lambert_attenuation(alpha)
        raise NotImplementedError("Optics attenuation model is not implemented yet.")

    def get_attenuation(self, frequency, electromagnetic_wave_type):
        if electromagnetic_wave_type == 'THz':
            return self.get_thz_attenuation(frequency)

        if electromagnetic_wave_type == 'Optics':
            return self.get_optics_attenuation(frequency)

        raise ValueError("Unsupported electromagnetic wave type.")

channel_model = ChannelModel(distance_m=distance_m)

def received_signal_power(transmitted_power, frequency, electromagnetic_wave_type='THz'):
    attenuation = channel_model.get_attenuation(frequency, electromagnetic_wave_type=electromagnetic_wave_type)
    return transmitted_power * attenuation

def get_photocurrent(received_power, responsivity=0.5):
    '''Convert received optical power to photocurrent using the photodetector's responsivity.'''
    return received_power * responsivity

chosen_frequency = 0.3

channel_model = ChannelModel(distance_m=distance_m)
channel_model.get_thz_attenuation(chosen_frequency)
print(f"Attenuation at {chosen_frequency} THz over {distance_m*100} cm: {channel_model.get_thz_attenuation(chosen_frequency)}")

for power_value in power:
    recieved_power = received_signal_power(transmitted_power=power_value, frequency=chosen_frequency, electromagnetic_wave_type='THz')
    print(f"Received signal power at {chosen_frequency} THz with transmitted power of {power_value} W: {recieved_power} W")



