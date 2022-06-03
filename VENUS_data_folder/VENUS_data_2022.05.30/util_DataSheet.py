#!/bin/env python3

import inspect
import itertools
import time
import types
import signal
import numpy as np
import datetime

import venus_utils.venusplc as venusplc


venus = venusplc.VENUSController('/home/damon/venus_data_project/config/config.ini')

def datasheet(writefile):
  pres=0.0; pres2=0.0
  extI=0.0; extI2=0.0
  for i in range(50):
    prespt=venus.read(['inj_mbar'])
    extpt=venus.read(['extraction_i'])
    pres=pres+prespt; pres2=pres2+prespt*prespt
    extI=extI+extpt; extI2=extI2+extpt*extpt
    time.sleep(0.25)
  writefile.write("%8.2e %8.2e\n"%(pres/50.0,np.sqrt(pres2/50.-pres*pres/(2500.))))
  writefile.write("%8.2f %8.2f\n"%(extI/50.0,np.sqrt(extI2/50.-extI*extI/(2500.))))
  # line 2: average inj pressure, rms inj pressure
  # line 3: average drain current, rms


  # Solenoids.  Line 4: inj, ext, mid, and sext currents
  writefile.write("%8.2f "%(venus.read(['inj_i'])))
  writefile.write("%8.2f "%(venus.read(['ext_i'])))
  writefile.write("%8.2f "%(venus.read(['mid_i'])))
  writefile.write("%8.2f\n"%(venus.read(['sext_i'])))

  # Extraction. Line 5: extraction voltage, drain current
  writefile.write("%8.2f "%(venus.read(['extraction_v'])))
  writefile.write("%8.2f\n"%(venus.read(['extraction_i'])))

  # Puller. Line 6: puller voltage, current, and gap
  writefile.write("%8.2f "%(venus.read(['puller_v'])))
  writefile.write("%8.2f "%(venus.read(['puller_i'])))
  writefile.write("%8.2f\n"%(venus.read(['puller_raw_gap'])))

  # RF.  Line 7: 28 power, 18 power, 18 reflected
  writefile.write("%8.2f "%(venus.read(['g28_fw'])))
  writefile.write("%8.2f "%(venus.read(['k18_fw'])))
  writefile.write("%8.2f\n"%(venus.read(['k18_ref'])))

  # Bias  Line 8: bias voltage, current
  writefile.write("%8.2f "%(venus.read(['bias_v'])))
  writefile.write("%8.2f\n"%(venus.read(['bias_i'])))

  # Vacuum  Line 9: injection pressure, extraction pressure, BL pressure
  writefile.write("%8.2e "%(venus.read(['inj_mbar'])))
  writefile.write("%8.2e "%(venus.read(['ext_mbar'])))
  writefile.write("%8.2e\n"%(venus.read(['bl_mig2_mbar'])))

  # Gas Valves. Line 10: balzer settings (1,2,5,6,7)
  writefile.write("%8.2f "%(venus.read(['gas_balzer_1'])))
  writefile.write("%8.2f "%(venus.read(['gas_balzer_2'])))
  writefile.write("%8.2f "%(venus.read(['gas_balzer_5'])))
  writefile.write("%8.2f "%(venus.read(['gas_balzer_6'])))
  writefile.write("%8.2f\n"%(venus.read(['gas_balzer_7'])))


  # Gas Names.  Line 11.  balzer gas names (1,2,5,6,7)
  writefile.write("%2i "%(venus.read(['gas_name_1'])))
  writefile.write("%2i "%(venus.read(['gas_name_2'])))
  writefile.write("%2i "%(venus.read(['gas_name_5'])))
  writefile.write("%2i "%(venus.read(['gas_name_6'])))
  writefile.write("%2i\n"%(venus.read(['gas_name_7'])))

  #  HT Oven. Line 12. ht oven voltage, current
  writefile.write("%8.2f "%(venus.read(['ht_oven_v'])))
  writefile.write("%8.2f\n"%(venus.read(['ht_oven_i'])))
  
  # LT Oven.  Line 13: LT oven 1 set point, 2 set point, 1 temp, 2 temp
  writefile.write("%8.2f "%(venus.read(['lt_oven_1_sp'])))
  writefile.write("%8.2f "%(venus.read(['lt_oven_2_sp'])))
  writefile.write("%8.2f "%(venus.read(['lt_oven_1_temp'])))
  writefile.write("%8.2f\n"%(venus.read(['lt_oven_2_temp'])))

  # Glaser.  Line 14: glaser current, batman current
  writefile.write("%8.2f "%(venus.read(['glaser_1'])))
  # Beam-Line
  writefile.write("%8.2f\n"%(venus.read(['bl_robin_i'])))
  
  # X-Rays.  Line 15: xray source, xray exit
  writefile.write("%8.2f "%(venus.read(['x_ray_source'])))
  writefile.write("%8.2f\n"%(venus.read(['x_ray_exit'])))

  # Cryogenics. Line 16: LHe pressure, LHe level
  #                  17: cryo press, 4K heater power, 4k cold mass
  #                  18: 4K cyros (e,w,ne,nw)
  #                  19: 4K heat cond, i feedthrough, heater(K)
  #                  20: 50K bar, Ne, NW, shield bottom
  #                  21  bottom LN vessel, 70K cond bar
  writefile.write("%8.2f "%(venus.read(['LHe_psi'])))
  writefile.write("%8.2f\n"%(venus.read(['LHe_level_in'])))
  writefile.write("%8.2e "%(venus.read(['cryo_vac_torr'])))
  writefile.write("%8.2f "%(venus.read(['four_k_heater_power'])))
  writefile.write("%8.2f\n"%(venus.read(['four_k_cold_mass'])))
  writefile.write("%8.2f "%(venus.read(['four_k_cryo_e'])))
  writefile.write("%8.2f "%(venus.read(['four_k_cryo_w'])))
  writefile.write("%8.2f "%(venus.read(['four_k_cryo_ne'])))
  writefile.write("%8.2f\n"%(venus.read(['four_k_cryo_nw'])))
  writefile.write("%8.2f "%(venus.read(['four_k_heat_cond'])))
  writefile.write("%8.2f "%(venus.read(['four_k_i_feedthrough'])))
  writefile.write("%8.2f\n"%(venus.read(['four_k_heater_k'])))
  writefile.write("%8.2f "%(venus.read(['fifty_k_cond_bar'])))
  writefile.write("%8.2f "%(venus.read(['fifty_k_cond_bar_NE'])))
  writefile.write("%8.2f "%(venus.read(['fifty_k_cond_bar_NW'])))
  writefile.write("%8.2f\n"%(venus.read(['fifty_k_shield_bot'])))
  writefile.write("%8.2f "%(venus.read(['bottom_ln_vessel'])))
  writefile.write("%8.2f\n"%(venus.read(['seventy_k_cond_bar'])))


filename = 'datasheet_'+str(int(time.time()))
writefile=open(filename,'w')
datasheet(writefile)
writefile.close()
