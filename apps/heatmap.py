#!/usr/bin/env python
# coding: utf-8

# In[20]:


# ee.Initialize()
# Map = geemap.Map()


# #                                VMRC DROUGHT RESISTANCE TOOL (V.1.0.0)

# In[1]:


import streamlit as st
import leafmap.foliumap as leafmap
import ee
import geemap
import ipyleaflet
from ipyleaflet import Map, basemaps

# Imports for JupyterLite

import piplite


import ipywidgets as widgets

import os
import ee
import geemap
import ipywidgets as widgets
from bqplot import pyplot as plt
from ipyleaflet import WidgetControl
import subprocess
import geemap.colormaps as cm
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import Output
from ipyleaflet import WidgetControl


import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:





    # In[3]:
def app():

    syear = widgets.IntSlider(min=1900, max=2099, value=2010,description='YEAR:',width='5000px')
    smonth = widgets.IntSlider(min=1, max=1, value=1)
    sday = widgets.IntSlider(min=1, max=1, value=1)
    uis = widgets.HBox([syear])
    def s(syear,smonth,sday):
       print((syear,smonth,sday)) 
    start=widgets.interactive_output(s, {'syear':syear,'smonth':smonth,'sday':sday})
    # display(uis,start)


    # In[4]:


    eyear = widgets.IntSlider(min=1900, max=2099, value=2022)
    emonth = widgets.IntSlider(min=12, max=12, value=12)
    eday = widgets.IntSlider(min=28, max=30, value=30)
    uie = widgets.HBox([syear,emonth,eday])
    def e(syear,emonth,eday):
       print((syear,emonth,eday)) 
    end=widgets.interactive_output(e, {'syear':syear,'emonth':emonth,'eday':eday})
    # display(uie,end)


    # In[5]:


    end_year_widget = widgets.IntSlider(min=1900, max=2100, value=2010, description='End year:', width=400)
    end_month_widget = widgets.IntSlider(min=1, max=12, value=1, description='End Month:', width=400)
    end_day_widget = widgets.IntSlider(min=29, max=30, value=30, description='End Day:', width=400)


    # In[6]:


    center = [45,-483]
    zoom = 5
    Map = geemap.Map(basemap=basemaps.Esri.WorldImagery, center=center, zoom=zoom)

    database = widgets.Dropdown(
        value = 'Real-Time Dataset (GRIDMET)',
        options = [None,'Real-Time Dataset (GRIDMET)'],
        description='Real-Time Dataset :',
        style = {'description_width':'initial'},
        layout = widgets.Layout(width = '500px'),
        button_style = 'primary'
    )

    scenario = widgets.Dropdown(
        value = None,
        options = [None,'rcp45','rcp85'],
        description='Climate Scenarios :',
        style = {'description_width':'initial'},
        layout = widgets.Layout(width = '500px')
    )

    model = widgets.Dropdown(
        value = None,
        options = [None,'BNU-ESM','CCSM4','CNRM-CM5','CSIRO-Mk3-6-0','CanESM2','GFDL-ESM2G','HadGEM2-CC365','IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM5B-LR','MIROC-ESM-CHEM','MIROC-ESM',
                   'MIROC5','MRI-CGCM3','NorESM1-M','bcc-csm1-1-m','bcc-csm1-1','inmcm4'],
        description='Climate Models :',
        style = {'description_width':'initial'},
        layout = widgets.Layout(width = '500px')

    )


    # Add an output widget to the map
    output_widget = widgets.Output(layout={'border': '1px solid black'})
    output_control = WidgetControl(widget=output_widget, position='bottomright')
    Map.add_control(output_control)

    latitude= widgets.FloatText(
        value=-123.81,
        description='latitude:',
        disabled=False
    )
    longitute=widgets.FloatText(
        value=44.53,
        description='longitute:',
        disabled=False
    )

    widgets.HBox([longitute, latitude])


    climate = widgets.Dropdown(
        value = None,
        options = ['Maximum Temperature', 'Maximum VPD'],
        description = 'Climate Variables:',
        style = {'description_width': 'initial'},
        layout = widgets.Layout(width = '500px')
    )


    mdwp_df_condition = widgets.Dropdown(
        value = None,
        options = ['Wet', 'Intermediate','Dry','Extreme_Dry'],
        description = 'Mid-Day Water Potential Douglas-fir:',
        style = {'description_width': 'initial'},
        layout = widgets.Layout(width = '550px')
    )

    mdwp_wh_condition = widgets.Dropdown(
        value = None,
        options = ['Wet', 'Intermediate','Dry','Extreme_Dry'],
        description = 'Mid-Day Water Potential Western Hemlock:',
        style = {'description_width': 'initial'},
        layout = widgets.Layout(width = '550px')
    )

    plc_df = widgets.Dropdown(
        value = None,
        options = ['Y10 # Wet Soil Condition','Y20 # Wet Soil Condition','Y30 # Wet Soil Condition','Y40 # Wet Soil Condition','Y50 # Wet Soil Condition','Y60 # Wet Soil Condition',
                  'Y10 # Intermediate Soil Condition','Y20 # Intermediate Soil Condition','Y30 # Intermediate Soil Condition','Y40 # Intermediate Soil Condition',
                  'Y50 # Intermediate Soil Condition','Y60 # Intermediate Soil Condition',
                  'Y10 # Dry Soil Condition','Y20 # Dry Soil Condition','Y30 # Dry Soil Condition','Y40 # Dry Soil Condition','Y50 # Dry Soil Condition','Y60 # Dry Soil Condition',
                  'Y10 # Extreme Dry Soil Condition','Y20 # Extreme Dry Soil Condition','Y30 # Extreme Dry Soil Condition','Y40 # Extreme Dry Soil Condition','Y50 # Extreme Dry Soil Condition',
                  'Y60 # Extreme Dry Soil Condition'],
        description = 'Percent Loss of Conductance - Douglas-fir :',
        style = {'description_width': 'initial' },
        layout = widgets.Layout(width = '550px')

    )

    plc_wh = widgets.Dropdown(
        value = None,
        options = ['Y50 # Wet Soil Condition','Y50 # Intermediate Soil Condition','Y50 # Dry Soil Condition','Y50 # Extreme Dry Soil Condition'],
        description = 'Percent Loss of Conductance - Western Hemlock :',
        style = {'description_width': 'initial' },
        layout = widgets.Layout(width = '550px')

    )

    btns = widgets.ToggleButtons(
        value = None,
        options = ['Apply', 'Reset'],
        button_style = 'primary',

    )



    btns_style_button_width = '180px'

    output = widgets.Output()

    # widgets.VBox([database,scenario,model,latitude,longitute,climate,mdwp_df_condition,mdwp_wh_condition,plc_df,plc_wh,btns,output,Map,start,end,output_widget])
    # box = widgets.VBox([Map,output_widget])

    # display(uis,start,end)



    # In[7]:


    from ipywidgets import Button, Layout, jslink, IntText, IntSlider

    def create_expanded_button(description, button_style):
        return Button(description=description, button_style=button_style, layout=Layout(height='auto', width='auto'))
    from ipywidgets import GridspecLayout

    grid = GridspecLayout(6, 5)

    for i in range(6):
        for j in range(5):
            grid[i, j] = create_expanded_button('Button {} - {}'.format(i, j), 'primary')

    grid = GridspecLayout(6, 5, height='200px')
    grid[0, 0] = uis
    grid[1, 0] = database
    grid[2, 0] = scenario
    grid[3, 0] = model
    grid[4, 0] = latitude
    grid[5, 0] = longitute
    grid[1,1] = climate
    grid[2,1] = mdwp_df_condition
    grid[3,1] = mdwp_wh_condition
    grid[1,2] = plc_df
    grid[2,2] = plc_wh
    grid[5,1] = btns
    # grid[:, 0] = create_expanded_button('Two', 'info')
    # grid[3, 1] = create_expanded_button('Three', 'warning')
    # grid[3, 2] = create_expanded_button('Four', 'danger')
    grid


    # In[8]:


    # Capture user interaction with the map
    def handle_interaction(**kwargs):
        latlon = kwargs.get('coordinates')
        if kwargs.get('type') == 'click':
            Map.default_style = {'cursor': 'wait'}
            # xy = ee.Geometry.Point(latlon[::-1])

            with output_widget:
                output_widget.clear_output()
                print(latlon)
        Map.default_style = {'cursor': 'pointer'}
    

    Map.on_interaction(handle_interaction)


    # In[9]:


    def database_change(change):
        if change['new']:
            with output:
                output.clear_output()
                print(change['new'])
    database.observe(database_change,'value')

    def scenario_change(change):
        if change['new']:
            with output:
                output.clear_output()
                print(change['new'])
    scenario.observe(scenario_change,'value')


    def model_change(change):
        if change['new']:
            with output:
                output.clear_output()
                print(change['new'])
    model.observe(model_change,'value')

    def climate_change(change):
        if change['new']:
            with output:
                output.clear_output()
                print(change['new'])
    climate.observe(climate_change,'value')

    ### DOUGLAS - FIR

    def mdwp_df_condition_change(change):
        if change['new']:
            with output:
                output.clear_output()
                print(change['new'])
    mdwp_df_condition.observe(mdwp_df_condition_change,'value')

    def plc_df_change(change):
        if change['new']:
            with output:
                output.clear_output()
                print(change['new'])
    plc_df.observe(plc_df_change,'value')


    #### WESTERN HEMLOCK

    def mdwp_wh_condition_change(change):
        if change['new']:
            with output:
                output.clear_output()
                print(change['new'])
    mdwp_wh_condition.observe(mdwp_wh_condition_change,'value')

    def plc_wh_change(change):
        if change['new']:
            with output:
                output.clear_output()
                print(change['new'])
    plc_wh.observe(plc_wh_change,'value')


    # In[10]:


    def btns_button_click(change):
        with output:
            output.clear_output()
            oregon = ee.FeatureCollection('TIGER/2018/States').filter(ee.Filter.eq('STATEFP', '41'))
            washington = ee.FeatureCollection('TIGER/2018/States').filter(ee.Filter.eq('STATEFP', '53'))
            states = oregon.merge(washington)
            Map.centerObject(states)


            start_date = '{}'.format(str(syear.value)+'-'+str(smonth.value)+'-'+str(sday.value))
            end_date = '{}'.format(str(syear.value)+'-'+str(emonth.value)+'-'+str(eday.value))

            if scenario.value is not None: 
                db = (ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA').filter(ee.Filter.date(start_date, end_date)).filterMetadata('scenario','equals',scenario.value).filterMetadata('model','equals',model.value).select('tasmax','rhsmin'))
            if database.value is database.options[1]:
                db = (ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filter(ee.Filter.date(start_date, end_date)).select('tmmx','rmin'))
                def conv_tmmx (img):
                    tmmx = img.expression(
                        'con_tmmx * 1',
                        {'con_tmmx': img.select('tmmx')
                        }).rename('tasmax')
                    return img.addBands(tmmx)
                def conv_rmin (img):
                    rmin = img.expression(
                        'con_rmin * 1',
                        {'con_rmin': img.select('rmin')
                        }).rename('rhsmin')
                    return img.addBands(rmin)
                db = db.map(conv_tmmx)
                db = db.map(conv_rmin)
                
                # db = (ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA').filter(ee.Filter.date(start_date, end_date)).filterMetadata('scenario','equals','rcp45')
                #         .filterMetadata('model','equals','BNU-ESM')
                #         .select('tasmax','rhsmin'))
            
            dataset = db.mean()
        
            palette = cm.palettes.seismic
        
            def func_uwl (img):
                TempMaxCelcius = img.expression(
                      'tmmxe - 273.15',
                      {'tmmxe': img.select('tasmax')
                      }).rename('tmaxcel').clip(states)

                return img.addBands(TempMaxCelcius)


            dataset = db.map(func_uwl)
            tmaxcel = dataset.select('tmaxcel')
            vis_tmmx= {
                    'min': 0,
                    'max': 55,
                    'palette': palette,
                    }
            colors_tmmx = vis_tmmx['palette']
            vmin_tmmx = vis_tmmx['min']
            vmax_tmmx = vis_tmmx['max']
        

        
            def func_vpsa (img):
                vpsatt = img.expression(
                      '6.11 * exp((0.0000025 / 461) * (1 / 273 - 1 / (273 + tmmxCelcius)))',
                      {'tmmxCelcius': img.select('tmaxcel')
                      }).rename('vpsat')

                return img.addBands(vpsatt)


            dataset = dataset.map(func_vpsa)
            vp = dataset.select('vpsat')

            def func_vpdmax (img):
                vpdmax = img.expression(
                      '((100-rmin)/100)*vpsat',
                      {'rmin': img.select('rhsmin'),
                       'vpsat': img.select('vpsat')
                      }).rename('vpdmax').clip(states)

                return img.addBands(vpdmax)


            dataset = dataset.map(func_vpdmax)
            vpdmax = dataset.select('vpdmax')
            vis_vpsat= {
                    'min': 0,
                    'max': 6,
                    'palette': palette,
                    }
            colors_vpsat = vis_vpsat['palette']
            vmin_vpsat = vis_vpsat['min']
            vmax_vpsat = vis_vpsat['max']
        
            def func_whmdwp_wet (img):
                whmdwp_wet = img.expression(
                      '0.4615*(5**0.2204)*(vpd**-0.1999)*(tmax**1.0133)',
                      {'tmax': img.select('tmaxcel'),
                       'vpd': img.select('vpdmax')
                      }).rename('whmdwp_wet')

                return img.addBands(whmdwp_wet)

            def func_whmdwp_intermediate (img):
                whmdwp_intermediate = img.expression(
                      '0.4615*(10**0.2204)*(vpd**-0.1999)*(tmax**1.0133)',
                      {'tmax': img.select('tmaxcel'),
                       'vpd': img.select('vpdmax')
                      }).rename('whmdwp_intermediate')

                return img.addBands(whmdwp_intermediate)

            def func_whmdwp_dry (img):
                whmdwp_dry = img.expression(
                      '0.4615*(15**0.2204)*(vpd**-0.1999)*(tmax**1.0133)',
                      {'tmax': img.select('tmaxcel'),
                       'vpd': img.select('vpdmax')
                      }).rename('whmdwp_dry')

                return img.addBands(whmdwp_dry)

            def func_whmdwp_extreme_dry (img):
                whmdwp_extreme_dry = img.expression(
                      '0.4615*(20**0.2204)*(vpd**-0.1999)*(tmax**1.0133)',
                      {'tmax': img.select('tmaxcel'),
                       'vpd': img.select('vpdmax')
                      }).rename('whmdwp_extreme_dry')

                return img.addBands(whmdwp_extreme_dry)


            dataset = dataset.map(func_whmdwp_wet)
            dataset = dataset.map(func_whmdwp_intermediate)
            dataset = dataset.map(func_whmdwp_dry)
            dataset = dataset.map(func_whmdwp_extreme_dry)

            whmdwp_wet = dataset.select('whmdwp_wet')
            whmdwp_intermediate = dataset.select('whmdwp_intermediate')
            whmdwp_dry = dataset.select('whmdwp_dry')
            whmdwp_extreme_dry = dataset.select('whmdwp_extreme_dry')
            vis_soil_condition= {
                    'min': 5,
                    'max': 20,
                    'palette': palette,
                    }
            colors_soil_condition = vis_soil_condition['palette']
            vmin_soil_condition = vis_soil_condition['min']
            vmax_soil_condition = vis_soil_condition['max']
        
            def func_dfmdwp_wet (img):
                dfmdwp_wet = img.expression(
                      '2.1964*(5**0.4245)*(vpd**0.0196)*(tmax**0.3648)',
                      {'tmax': img.select('tmaxcel'),
                       'vpd': img.select('vpdmax')
                      }).rename('dfmdwp_wet')

                return img.addBands(dfmdwp_wet)

            def func_dfmdwp_intermediate (img):
                dfmdwp_intermediate = img.expression(
                      '2.1964*(10**0.4245)*(vpd**0.0196)*(tmax**0.3648)',
                      {'tmax': img.select('tmaxcel'),
                       'vpd': img.select('vpdmax')
                      }).rename('dfmdwp_int')

                return img.addBands(dfmdwp_intermediate)

            def func_dfmdwp_dry (img):
                dfmdwp_dry = img.expression(
                      '2.1964*(15**0.4245)*(vpd**0.0196)*(tmax**0.3648)',
                      {'tmax': img.select('tmaxcel'),
                       'vpd': img.select('vpdmax')
                      }).rename('dfmdwp_dry')

                return img.addBands(dfmdwp_dry)

            def func_dfmdwp_extreme_dry (img):
                dfmdwp_extreme_dry = img.expression(
                      '2.1964*(20**0.4245)*(vpd**0.0196)*(tmax**0.3648)',
                      {'tmax': img.select('tmaxcel'),
                       'vpd': img.select('vpdmax')
                      }).rename('dfmdwp_extreme_dry')

                return img.addBands(dfmdwp_extreme_dry)


            dataset = dataset.map(func_dfmdwp_wet)
            dataset = dataset.map(func_dfmdwp_intermediate)
            dataset = dataset.map(func_dfmdwp_dry)
            dataset = dataset.map(func_dfmdwp_extreme_dry)

            dfmdwp_wet = dataset.select('dfmdwp_wet')
            dfmdwp_int = dataset.select('dfmdwp_int')
            dfmdwp_dry = dataset.select('dfmdwp_dry')
            dfmdwp_extreme_dry = dataset.select('dfmdwp_extreme_dry')
        
            def func_whplc_wet_five (img):
                whplc_wet_five = img.expression(
                      '100 / (1 + exp(-0.408000922128669 * ((whmdwp_wet / 10) - 5)))',
                      {'whmdwp_wet': img.select('whmdwp_wet')
                      }).rename('whplc_wet_five')

                return img.addBands(whplc_wet_five)
            dataset = dataset.map(func_whplc_wet_five)
            whplc_wet_five = dataset.select('whplc_wet_five')

            def func_whplc_intermediate_five (img):
                whplc_intermediate_five = img.expression(
                      '100 / (1 + exp(-0.408000922128669 * ((whmdwp_intermediate / 10) - 5)))',
                      {'whmdwp_intermediate': img.select('whmdwp_intermediate')
                      }).rename('whplc_intermediate_five')

                return img.addBands(whplc_intermediate_five)

            dataset = dataset.map(func_whplc_intermediate_five)
            whplc_intermediate_five = dataset.select('whplc_intermediate_five')


            def func_whplc_dry_five (img):
                whplc_dry_five = img.expression(
                      '100 / (1 + exp(-0.408000922128669 * ((whmdwp_dry / 10) - 5)))',
                      {'whmdwp_dry': img.select('whmdwp_dry')
                      }).rename('whplc_dry_five')

                return img.addBands(whplc_dry_five)

            dataset = dataset.map(func_whplc_dry_five)
            whplc_dry_five = dataset.select('whplc_dry_five')


            def func_whplc_extreme_dry_five (img):
                whplc_extreme_dry_five = img.expression(
                      '100 / (1 + exp(-0.408000922128669 * ((whplc_extreme_dry_five / 10) - 5)))',
                      {'whplc_extreme_dry_five': img.select('whmdwp_extreme_dry')
                      }).rename('whplc_extreme_dry_five')

                return img.addBands(whplc_extreme_dry_five)

            dataset = dataset.map(func_whplc_extreme_dry_five)
            whplc_extreme_dry_five = dataset.select('whplc_extreme_dry_five')   
        
            vis_plc= {
                            'min': 0,
                            'max': 100,
                            'palette': palette,
                            }
            colors_plc = vis_plc['palette']
            vmin_plc = vis_plc['min']
            vmax_plc = vis_plc['max']        
        
            def func_dfplc_one (img):
                dfplc_one = img.expression(
                      '100 / (1 + exp(-1.63855557010636 * ((dfmdwp_wet / 10) - 1)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_one')

                return img.addBands(dfplc_one)

            def func_dfplc_one5 (img):
                dfplc_one5 = img.expression(
                      '100 / (1 + exp(-1.69944826381321 * ((dfmdwp_wet / 10) - 1.5)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_one5')

                return img.addBands(dfplc_one5)

            def func_dfplc_two (img):
                dfplc_two = img.expression(
                      '100 / (1 + exp(-1.63016863612044 * ((dfmdwp_wet / 10) - 2)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_two')

                return img.addBands(dfplc_two)

            def func_dfplc_two5 (img):
                dfplc_two5 = img.expression(
                      '100 / (1 + exp(-1.43895958154504 * ((dfmdwp_wet / 10) - 2.5)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_two5')

                return img.addBands(dfplc_two5)

            def func_dfplc_three (img):
                dfplc_three = img.expression(
                      '100 / (1 + exp(-1.18402632453238 * ((dfmdwp_wet / 10) - 3)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_three')

                return img.addBands(dfplc_three)

            def func_dfplc_three5 (img):
                dfplc_three5 = img.expression(
                      '100 / (1 + exp(-0.929094196206154 * ((dfmdwp_wet / 10) - 3.5)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_three5')

                return img.addBands(dfplc_three5)

            def func_dfplc_four (img):
                dfplc_four = img.expression(
                      '100 / (1 + exp(-0.710710785489621 * ((dfmdwp_wet / 10) - 4)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_four')

                return img.addBands(dfplc_four)

            def func_dfplc_four5 (img):
                dfplc_four5 = img.expression(
                      '100 / (1 + exp(-0.538477163750487 * ((dfmdwp_wet / 10) - 4.5)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_four5')

                return img.addBands(dfplc_four5)

            def func_dfplc_five (img):
                dfplc_five = img.expression(
                      '100 / (1 + exp(-0.408000922128669 * ((dfmdwp_wet / 10) - 5)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_five')

                return img.addBands(dfplc_five)

            def func_dfplc_five5 (img):
                dfplc_five5 = img.expression(
                      '100 / (1 + exp(-0.310727450746565 * ((dfmdwp_wet / 10) - 5.5)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_five5')

                return img.addBands(dfplc_five5)

            def func_dfplc_six (img):
                dfplc_six = img.expression(
                      '100 / (1 + exp(-0.238404309312474 * ((dfmdwp_wet / 10) - 6)))',
                      {'dfmdwp_wet': img.select('dfmdwp_wet')
                      }).rename('dfplc_six')

                return img.addBands(dfplc_six)


            dataset = dataset.map(func_dfplc_one)
            dataset = dataset.map(func_dfplc_one5)
            dataset = dataset.map(func_dfplc_two)
            dataset = dataset.map(func_dfplc_two5)
            dataset = dataset.map(func_dfplc_three)
            dataset = dataset.map(func_dfplc_three5)
            dataset = dataset.map(func_dfplc_four)
            dataset = dataset.map(func_dfplc_four5)
            dataset = dataset.map(func_dfplc_five)
            dataset = dataset.map(func_dfplc_five5)
            dataset = dataset.map(func_dfplc_six)

            dfplc_one = dataset.select('dfplc_one')
            dfplc_one5 = dataset.select('dfplc_one5')
            dfplc_two = dataset.select('dfplc_two')
            dfplc_two5 = dataset.select('dfplc_two5')
            dfplc_three = dataset.select('dfplc_three')
            dfplc_three5 = dataset.select('dfplc_three5')
            dfplc_four = dataset.select('dfplc_four')
            dfplc_four5 = dataset.select('dfplc_four5')
            dfplc_five = dataset.select('dfplc_five')
            dfplc_five5 = dataset.select('dfplc_five5')
            dfplc_six = dataset.select('dfplc_six')


            def func_dfplc_intermadiate_one (img):
                dfplc_intermadiate_one = img.expression(
                      '100 / (1 + exp(-1.63855557010636 * ((dfmdwp_int / 10) - 1)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_one')

                return img.addBands(dfplc_intermadiate_one)

            def func_dfplc_intermadiate_one5 (img):
                dfplc_intermadiate_one5 = img.expression(
                      '100 / (1 + exp(-1.69944826381321 * ((dfmdwp_int / 10) - 1.5)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_one5')

                return img.addBands(dfplc_intermadiate_one5)

            def func_dfplc_intermadiate_two (img):
                dfplc_intermadiate_two = img.expression(
                      '100 / (1 + exp(-1.63016863612044 * ((dfmdwp_int / 10) - 2)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_two')

                return img.addBands(dfplc_intermadiate_two)

            def func_dfplc_intermadiate_two5 (img):
                dfplc_intermadiate_two5 = img.expression(
                      '100 / (1 + exp(-1.43895958154504 * ((dfmdwp_int / 10) - 2.5)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_two5')

                return img.addBands(dfplc_intermadiate_two5)

            def func_dfplc_intermadiate_three (img):
                dfplc_intermadiate_three = img.expression(
                      '100 / (1 + exp(-1.18402632453238 * ((dfmdwp_int / 10) - 3)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_three')

                return img.addBands(dfplc_intermadiate_three)

            def func_dfplc_intermadiate_three5 (img):
                dfplc_intermadiate_three5 = img.expression(
                      '100 / (1 + exp(-0.929094196206154 * ((dfmdwp_int / 10) - 3.5)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_three5')

                return img.addBands(dfplc_intermadiate_three5)

            def func_dfplc_intermadiate_four (img):
                dfplc_intermadiate_four = img.expression(
                      '100 / (1 + exp(-0.710710785489621 * ((dfmdwp_int / 10) - 4)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_four')

                return img.addBands(dfplc_intermadiate_four)

            def func_dfplc_intermadiate_four5 (img):
                dfplc_intermadiate_four5 = img.expression(
                      '100 / (1 + exp(-0.538477163750487 * ((dfmdwp_int/ 10) - 4.5)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_four5')

                return img.addBands(dfplc_intermadiate_four5)

            def func_dfplc_intermadiate_five (img):
                dfplc_intermadiate_five = img.expression(
                      '100 / (1 + exp(-0.408000922128669 * ((dfmdwp_int / 10) - 5)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_five')

                return img.addBands(dfplc_intermadiate_five)

            def func_dfplc_intermadiate_five5 (img):
                dfplc_intermadiate_five5 = img.expression(
                      '100 / (1 + exp(-0.310727450746565 * ((dfmdwp_int / 10) - 5.5)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_five5')

                return img.addBands(dfplc_intermadiate_five5)

            def func_dfplc_intermadiate_six (img):
                dfplc_intermadiate_six = img.expression(
                      '100 / (1 + exp(-0.238404309312474 * ((dfmdwp_int / 10) - 6)))',
                      {'dfmdwp_int': img.select('dfmdwp_int')
                      }).rename('dfplc_intermediate_six')

                return img.addBands(dfplc_intermadiate_six)


            dataset = dataset.map(func_dfplc_intermadiate_one)
            dataset = dataset.map(func_dfplc_intermadiate_one5)
            dataset = dataset.map(func_dfplc_intermadiate_two)
            dataset = dataset.map(func_dfplc_intermadiate_two5)
            dataset = dataset.map(func_dfplc_intermadiate_three)
            dataset = dataset.map(func_dfplc_intermadiate_three5)
            dataset = dataset.map(func_dfplc_intermadiate_four)
            dataset = dataset.map(func_dfplc_intermadiate_four5)
            dataset = dataset.map(func_dfplc_intermadiate_five)
            dataset = dataset.map(func_dfplc_intermadiate_five5)
            dataset = dataset.map(func_dfplc_intermadiate_six)

            dfplc_intermediate_one = dataset.select('dfplc_intermediate_one')
            dfplc_intermediate_one5 = dataset.select('dfplc_intermediate_one5')
            dfplc_intermediate_two = dataset.select('dfplc_intermediate_two')
            dfplc_intermediate_two5 = dataset.select('dfplc_intermediate_two5')
            dfplc_intermediate_three = dataset.select('dfplc_intermediate_three')
            dfplc_intermediate_three5 = dataset.select('dfplc_intermediate_three5')
            dfplc_intermediate_four = dataset.select('dfplc_intermediate_four')
            dfplc_intermediate_four5 = dataset.select('dfplc_intermediate_four5')
            dfplc_intermediate_five = dataset.select('dfplc_intermediate_five')
            dfplc_intermediate_five5 = dataset.select('dfplc_intermediate_five5')
            dfplc_intermediate_six = dataset.select('dfplc_intermediate_six')

            def func_dfplc_dry_one (img):
                dfplc_dry_one = img.expression(
                      '100 / (1 + exp(-1.63855557010636 * ((dfmdwp_dry / 10) - 1)))',
                      {'dfmdwp_dry': img.select('dfmdwp_dry')
                      }).rename('dfplc_dry_one')

                return img.addBands(dfplc_dry_one)

            def func_dfplc_dry_one5 (img):
                dfplc_dry_one5 = img.expression(
                      '100 / (1 + exp(-1.69944826381321 * ((dfmdwp_dry / 10) - 1.5)))',
                      {'dfmdwp_dry': img.select('dfmdwp_dry')
                      }).rename('dfplc_dry_one5')

                return img.addBands(dfplc_dry_one5)

            def func_dfplc_dry_two (img):
                dfplc_dry_two = img.expression(
                      '100 / (1 + exp(-1.63016863612044 * ((dfmdwp_dry / 10) - 2)))',
                      {'dfmdwp_dry': img.select('dfmdwp_dry')
                      }).rename('dfplc_dry_two')

                return img.addBands(dfplc_dry_two)

            def func_dfplc_dry_two5 (img):
                dfplc_dry_two5 = img.expression(
                      '100 / (1 + exp(-1.43895958154504 * ((dfmdwp_dry / 10) - 2.5)))',
                      {'dfmdwp_dry': img.select('dfmdwp_dry')
                      }).rename('dfplc_dry_two5')

                return img.addBands(dfplc_dry_two5)

            def func_dfplc_dry_three (img):
                dfplc_dry_three = img.expression(
                      '100 / (1 + exp(-1.18402632453238 * ((dfmdwp_dry / 10) - 3)))',
                      {'dfmdwp_dry': img.select('dfmdwp_dry')
                      }).rename('dfplc_dry_three')

                return img.addBands(dfplc_dry_three)

            def func_dfplc_dry_three5 (img):
                dfplc_dry_three5 = img.expression(
                      '100 / (1 + exp(-0.929094196206154 * ((dfmdwp_dry / 10) - 3.5)))',
                      {'dfmdwp_dry': img.select('dfmdwp_int')
                      }).rename('dfplc_dry_three5')

                return img.addBands(dfplc_dry_three5)

            def func_dfplc_dry_four (img):
                dfplc_dry_four = img.expression(
                      '100 / (1 + exp(-0.710710785489621 * ((dfmdwp_dry / 10) - 4)))',
                      {'dfmdwp_dry': img.select('dfmdwp_dry')
                      }).rename('dfplc_dry_four')

                return img.addBands(dfplc_dry_four)

            def func_dfplc_dry_four5 (img):
                dfplc_dry_four5 = img.expression(
                      '100 / (1 + exp(-0.538477163750487 * ((dfmdwp_dry/ 10) - 4.5)))',
                      {'dfmdwp_dry': img.select('dfmdwp_dry')
                      }).rename('dfplc_dry_four5')

                return img.addBands(dfplc_dry_four5)

            def func_dfplc_dry_five (img):
                dfplc_dry_five = img.expression(
                      '100 / (1 + exp(-0.408000922128669 * ((dfmdwp_dry / 10) - 5)))',
                      {'dfmdwp_dry': img.select('dfmdwp_dry')
                      }).rename('dfplc_dry_five')

                return img.addBands(dfplc_dry_five)

            def func_dfplc_dry_five5 (img):
                dfplc_dry_five5 = img.expression(
                      '100 / (1 + exp(-0.310727450746565 * ((dfmdwp_dry / 10) - 5.5)))',
                      {'dfmdwp_dry': img.select('dfmdwp_dry')
                      }).rename('dfplc_dry_five5')

                return img.addBands(dfplc_dry_five5)

            def func_dfplc_dry_six (img):
                dfplc_dry_six = img.expression(
                      '100 / (1 + exp(-0.238404309312474 * ((dfmdwp_dry / 10) - 6)))',
                      {'dfmdwp_dry': img.select('dfmdwp_dry')
                      }).rename('dfplc_dry_six')

                return img.addBands(dfplc_dry_six)


            dataset = dataset.map(func_dfplc_dry_one)
            dataset = dataset.map(func_dfplc_dry_one5)
            dataset = dataset.map(func_dfplc_dry_two)
            dataset = dataset.map(func_dfplc_dry_two5)
            dataset = dataset.map(func_dfplc_dry_three)
            dataset = dataset.map(func_dfplc_dry_three5)
            dataset = dataset.map(func_dfplc_dry_four)
            dataset = dataset.map(func_dfplc_dry_four5)
            dataset = dataset.map(func_dfplc_dry_five)
            dataset = dataset.map(func_dfplc_dry_five5)
            dataset = dataset.map(func_dfplc_dry_six)

            dfplc_dry_one = dataset.select('dfplc_dry_one')
            dfplc_dry_one5 = dataset.select('dfplc_dry_one5')
            dfplc_dry_two = dataset.select('dfplc_dry_two')
            dfplc_dry_two5 = dataset.select('dfplc_dry_two5')
            dfplc_dry_three = dataset.select('dfplc_dry_three')
            dfplc_dry_three5 = dataset.select('dfplc_dry_three5')
            dfplc_dry_four = dataset.select('dfplc_dry_four')
            dfplc_dry_four5 = dataset.select('dfplc_dry_four5')
            dfplc_dry_five = dataset.select('dfplc_dry_five')
            dfplc_dry_five5 = dataset.select('dfplc_dry_five5')
            dfplc_dry_six = dataset.select('dfplc_dry_six')

            def func_dfplc_extreme_dry_one (img):
                dfplc_extreme_dry_one = img.expression(
                      '100 / (1 + exp(-1.63855557010636 * ((dfmdwp_exdry / 10) - 1)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_one')

                return img.addBands(dfplc_extreme_dry_one)

            def func_dfplc_extreme_dry_one5 (img):
                dfplc_extreme_dry_one5 = img.expression(
                      '100 / (1 + exp(-1.69944826381321 * ((dfmdwp_exdry / 10) - 1.5)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_one5')

                return img.addBands(dfplc_extreme_dry_one5)

            def func_dfplc_extreme_dry_two (img):
                dfplc_extreme_dry_two = img.expression(
                      '100 / (1 + exp(-1.63016863612044 * ((dfmdwp_exdry / 10) - 2)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_two')

                return img.addBands(dfplc_extreme_dry_two)

            def func_dfplc_extreme_dry_two5 (img):
                dfplc_extreme_dry_two5 = img.expression(
                      '100 / (1 + exp(-1.43895958154504 * ((dfmdwp_exdry / 10) - 2.5)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_two5')

                return img.addBands(dfplc_extreme_dry_two5)

            def func_dfplc_extreme_dry_three (img):
                dfplc_extreme_dry_three = img.expression(
                      '100 / (1 + exp(-1.18402632453238 * ((dfmdwp_exdry / 10) - 3)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_three')

                return img.addBands(dfplc_extreme_dry_three)

            def func_dfplc_extreme_dry_three5 (img):
                dfplc_extreme_dry_three5 = img.expression(
                      '100 / (1 + exp(-0.929094196206154 * ((dfmdwp_exdry / 10) - 3.5)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_three5')

                return img.addBands(dfplc_extreme_dry_three5)

            def func_dfplc_extreme_dry_four (img):
                dfplc_extreme_dry_four = img.expression(
                      '100 / (1 + exp(-0.710710785489621 * ((dfmdwp_exdry / 10) - 4)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_four')

                return img.addBands(dfplc_extreme_dry_four)

            def func_dfplc_extreme_dry_four5 (img):
                dfplc_extreme_dry_four5 = img.expression(
                      '100 / (1 + exp(-0.538477163750487 * ((dfmdwp_exdry/ 10) - 4.5)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_four5')

                return img.addBands(dfplc_extreme_dry_four5)

            def func_dfplc_extreme_dry_five (img):
                dfplc_extreme_dry_five = img.expression(
                      '100 / (1 + exp(-0.408000922128669 * ((dfmdwp_exdry / 10) - 5)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_five')

                return img.addBands(dfplc_extreme_dry_five)

            def func_dfplc_extreme_dry_five5 (img):
                dfplc_extreme_dry_five5 = img.expression(
                      '100 / (1 + exp(-0.310727450746565 * ((dfmdwp_exdry / 10) - 5.5)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_five5')

                return img.addBands(dfplc_extreme_dry_five5)

            def func_dfplc_extreme_dry_six (img):
                dfplc_extreme_dry_six = img.expression(
                      '100 / (1 + exp(-0.238404309312474 * ((dfmdwp_exdry / 10) - 6)))',
                      {'dfmdwp_exdry': img.select('dfmdwp_extreme_dry')
                      }).rename('dfplc_extreme_dry_six')

                return img.addBands(dfplc_extreme_dry_six)


            dataset = dataset.map(func_dfplc_extreme_dry_one)
            dataset = dataset.map(func_dfplc_extreme_dry_one5)
            dataset = dataset.map(func_dfplc_extreme_dry_two)
            dataset = dataset.map(func_dfplc_extreme_dry_two5)
            dataset = dataset.map(func_dfplc_extreme_dry_three)
            dataset = dataset.map(func_dfplc_extreme_dry_three5)
            dataset = dataset.map(func_dfplc_extreme_dry_four)
            dataset = dataset.map(func_dfplc_extreme_dry_four5)
            dataset = dataset.map(func_dfplc_extreme_dry_five)
            dataset = dataset.map(func_dfplc_extreme_dry_five5)
            dataset = dataset.map(func_dfplc_extreme_dry_six)

            dfplc_extreme_dry_one = dataset.select('dfplc_extreme_dry_one')
            dfplc_extreme_dry_one5 = dataset.select('dfplc_extreme_dry_one5')
            dfplc_extreme_dry_two = dataset.select('dfplc_extreme_dry_two')
            dfplc_extreme_dry_two5 = dataset.select('dfplc_extreme_dry_two5')
            dfplc_extreme_dry_three = dataset.select('dfplc_extreme_dry_three')
            dfplc_extreme_dry_three5 = dataset.select('dfplc_extreme_dry_three5')
            dfplc_extreme_dry_four = dataset.select('dfplc_extreme_dry_four')
            dfplc_extreme_dry_four5 = dataset.select('dfplc_extreme_dry_four5')
            dfplc_extreme_dry_five = dataset.select('dfplc_extreme_dry_five')
            dfplc_extreme_dry_five5 = dataset.select('dfplc_extreme_dry_five5')
            dfplc_extreme_dry_six = dataset.select('dfplc_extreme_dry_six')

            if change['new'] == 'Apply':
           
  
                # ASSIGN CLIMATE VARIABLE
                if climate.value is None:
                    print('')
                if climate.value is climate.options[0]:
                    Map.addLayer(tmaxcel,vis_tmmx,'Maximum Temperature')
                    with output:
                        output.clear_output()
                        Map.add_colorbar_branca(colors=colors_tmmx, vmin=vmin_tmmx, vmax=vmax_tmmx)
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('tmaxcel')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('Mean of Max. Temperature',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Max. Temp',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()            
                    # output_control = WidgetControl(widget=output, position="bottomleft")
                    # Map.add_control(output_control)  

                
                if climate.value is climate.options[1]:
                    Map.addLayer(vpdmax,vis_vpsat,'Maximum VPD')
                    Map.add_colorbar_branca(colors=colors_vpsat, vmin=vmin_vpsat, vmax=vmax_vpsat)
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('vpdmax')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('Mean of Max. VPD',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Max. VPD',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()                      
            
                # ASSIGN SOIL MOISTURE CONDITIONS - DOUGLAS FIR
                
                if mdwp_df_condition.value is None:
                    print('')
                if mdwp_df_condition.value is mdwp_df_condition.options[0]:
                    Map.addLayer(dfmdwp_wet,vis_soil_condition,'Wet')
                    Map.add_colorbar_branca(colors=colors_soil_condition, vmin=vmin_soil_condition, vmax=vmax_soil_condition)
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfmdwp_wet')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('Douglas-fir Mid-day Wapor Pressure',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir Mid-day Wapor Pressure # WET SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()
                    
                    
                if mdwp_df_condition.value is mdwp_df_condition.options[1]:
                    Map.addLayer(dfmdwp_int,vis_soil_condition,'Intermediate')
                    Map.add_colorbar_branca(colors=colors_soil_condition, vmin=vmin_soil_condition, vmax=vmax_soil_condition)
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfmdwp_int')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('Douglas-fir Mid-day Wapor Pressure',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir Mid-day Wapor Pressure # INTERMEDIATE SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()    
                
                
                if mdwp_df_condition.value is mdwp_df_condition.options[2]:
                    Map.addLayer(dfmdwp_dry,vis_soil_condition,'Dry')
                    Map.add_colorbar_branca(colors=colors_soil_condition, vmin=vmin_soil_condition, vmax=vmax_soil_condition)
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfmdwp_dry')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('Douglas-fir Mid-day Wapor Pressure',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir Mid-day Wapor Pressure # DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()    
                
                
                if mdwp_df_condition.value is mdwp_df_condition.options[3]:
                    Map.addLayer(dfmdwp_extreme_dry,vis_soil_condition,'Extreme_Dry')
                    Map.add_colorbar_branca(colors=colors_soil_condition, vmin=vmin_soil_condition, vmax=vmax_soil_condition)
                
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfmdwp_extreme_dry')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('Douglas-fir Mid-day Wapor Pressure',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir Mid-day Wapor Pressure # EXTREME DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()    
                
                
                
                # ASSIGN SOIL MOISTURE CONDITION - WESTERN HEMLOCK
                if mdwp_wh_condition.value is None:
                    print('')
                if mdwp_wh_condition.value is mdwp_wh_condition.options[0]:
                    Map.addLayer(whmdwp_wet,vis_soil_condition,'Wet')
                    Map.add_colorbar_branca(colors=colors_soil_condition, vmin=vmin_soil_condition, vmax=vmax_soil_condition)
                
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('whmdwp_wet')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('Western Hemlock Mid-day Wapor Pressure',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Western Hemlock Mid-day Wapor Pressure # WET SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()    
                
                
                if mdwp_wh_condition.value is mdwp_wh_condition.options[1]:
                    Map.addLayer(whmdwp_intermediate,vis_soil_condition,'Intermediate')
                    Map.add_colorbar_branca(colors=colors_soil_condition, vmin=vmin_soil_condition, vmax=vmax_soil_condition)
                
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('whmdwp_intermediate')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('Western Hemlock Mid-day Wapor Pressure',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Western Hemlock Mid-day Wapor Pressure # INTERMEDIATE SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()   
                    
                    
                if mdwp_wh_condition.value is mdwp_wh_condition.options[2]:
                    Map.addLayer(whmdwp_dry,vis_soil_condition,'Dry')
                    Map.add_colorbar_branca(colors=colors_soil_condition, vmin=vmin_soil_condition, vmax=vmax_soil_condition)
                
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('whmdwp_dry')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('Western Hemlock Mid-day Wapor Pressure',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Western Hemlock Mid-day Wapor Pressure # DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()   
                    
                    
                if mdwp_wh_condition.value is mdwp_wh_condition.options[3]:
                    Map.addLayer(whmdwp_extreme_dry,vis_soil_condition,'Extreme_Dry')
                    Map.add_colorbar_branca(colors=colors_soil_condition, vmin=vmin_soil_condition, vmax=vmax_soil_condition)
                
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('whmdwp_extreme_dry')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('Western Hemlock Mid-day Wapor Pressure',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Western Hemlock Mid-day Wapor Pressure # EXTREME DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()   
                    
            

                # ASSIGN PLC FOR DF
            
                if plc_df.value is None:
                    print("Drought resistance for Douglas-fir has not been selected")
                if plc_df.value is plc_df.options[0]:
                    Map.addLayer(dfplc_one, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y10 # - WET SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)
                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')

                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_one')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y10',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Percent Loss of Conductance # WET SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show()    
                
                
                
                if plc_df.value is plc_df.options[1]:
                    Map.addLayer(dfplc_two, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y20 # - WET SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)
                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_two')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y20',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Percent Loss of Conductance # WET SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                
                if plc_df.value is plc_df.options[2]:
                    Map.addLayer(dfplc_three, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y30 # - WET SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_three')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y30',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Percent Loss of Conductance # WET SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                
                if plc_df.value is plc_df.options[3]:
                    Map.addLayer(dfplc_four, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y40 # - WET SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_four')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y40',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir Percent Loss of Conductance # WET SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[4]:
                    Map.addLayer(dfplc_five, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y50 # - WET SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_five')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y50',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # WET SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[5]:
                    Map.addLayer(dfplc_six, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y60 # - WET SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_six')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y60',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # WET SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[6]:
                    Map.addLayer(dfplc_intermediate_one, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y10 # - INTERMEDIATE SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_intermediate_one')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y60',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # WET SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[7]:
                    Map.addLayer(dfplc_intermediate_two, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y20 # - INTERMEDIATE SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_intermediate_two')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y20',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # INTERMADIATE SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[8]:
                    Map.addLayer(dfplc_intermediate_three, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y30 # - INTERMEDIATE SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_intermediate_three')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y30',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # INTERMADIATE SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[9]:
                    Map.addLayer(dfplc_intermediate_four, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y40 # - INTERMEDIATE SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_intermediate_four')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y40',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # INTERMADIATE SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[10]:
                    Map.addLayer(dfplc_intermediate_five, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y50 # - INTERMEDIATE SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_intermediate_five')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y50',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # INTERMADIATE SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[11]:
                    Map.addLayer(dfplc_intermediate_six, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y60 # - INTERMEDIATE SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_intermediate_six')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y60',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # INTERMADIATE SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[12]:
                    Map.addLayer(dfplc_dry_one, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y10 # - DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_dry_one')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y10',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[13]:
                    Map.addLayer(dfplc_dry_two, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y20 # - DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_dry_two')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y20',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[14]:
                    Map.addLayer(dfplc_dry_three, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y30 # - DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_dry_three')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y30',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[15]:
                    Map.addLayer(dfplc_dry_four, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y40 # - DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_dry_four')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y40',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[16]:
                    Map.addLayer(dfplc_dry_five, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y50 # - DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_dry_five')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y50',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[17]:
                    Map.addLayer(dfplc_dry_six, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y60 # - DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_dry_six')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y60',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[18]:
                    Map.addLayer(dfplc_extreme_dry_one, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y10 # - EXTREME DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)
 
                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_extreme_dry_one')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y10',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # EXTREME DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[19]:
                    Map.addLayer(dfplc_extreme_dry_two, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y20 # - EXTREME DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_extreme_dry_two')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y20',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # EXTREME DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[20]:
                    Map.addLayer(dfplc_extreme_dry_three, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y30 # - EXTREME DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_extreme_dry_three')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y30',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # EXTREME DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[21]:
                    Map.addLayer(dfplc_extreme_dry_four, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y40 # - EXTREME DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_extreme_dry_four')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y40',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # EXTREME DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[22]:
                    Map.addLayer(dfplc_extreme_dry_five, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y50 # - EXTREME DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_extreme_dry_five')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y50',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # EXTREME DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    
                if plc_df.value is plc_df.options[23]:
                    Map.addLayer(dfplc_extreme_dry_six, vis_plc,'DOUGLAS-FIR PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y60 # - EXTREME DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('dfplc_extreme_dry_six')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y60',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Douglas-fir  Percent Loss of Conductance # EXTREME DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    

                            # ASSIGN PLC FOR WESTERN HEMLOCK plc_wh whplc_dry_five
            
                if plc_wh.value is None:
                    print("Drought resistance for Western Hemlock has not been selected")
                if plc_wh.value is plc_wh.options[0]:
                    Map.addLayer(whplc_wet_five, vis_plc,'WESTERN HEMLOCK PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y50 # - WET SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('whplc_wet_five')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y50',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Western Hemlock  Percent Loss of Conductance # WET SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    

                if plc_wh.value is plc_wh.options[1]:
                    Map.addLayer(whplc_intermediate_five, vis_plc,'WESTERN HEMLOCK PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y50 # - INTERMEDIATE SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('whplc_intermediate_five')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y50',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Western Hemlock  Percent Loss of Conductance # INTERMEDIATE SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    

                if plc_wh.value is plc_wh.options[2]:
                    Map.addLayer(whplc_dry_five, vis_plc,'WESTERN HEMLOCK PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y50 # - DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('whplc_dry_five')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y50',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Western Hemlock  Percent Loss of Conductance # DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    

                if plc_wh.value is plc_wh.options[3]:
                    Map.addLayer(whplc_extreme_dry_five, vis_plc,'WESTERN HEMLOCK PERCENT LOSS CONDUCTANCE - DROUGHT RESISTANCE # Y50 # - EXTREME DRY SOIL CONTIDION')
                    Map.add_colorbar_branca(colors=colors_plc, vmin=vmin_plc, vmax=vmax_plc)

                    print('Species has been selected')
                    print('Climate variables has been selected')
                    print('Soil moisture condition has been selected')
                    print('Drought resistance has been selected')
                                                                                                                                                
                    poi = ee.Geometry.Point(latitude.value,longitute.value).buffer(5000)
                    def poi_mean(img):
                        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30).get('whplc_extreme_dry_five')
                        return img.set('date', img.date().format()).set('mean',mean)
                    poi_reduced_imgs = dataset.map(poi_mean)
                    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)
                    df = pd.DataFrame(nested_list.getInfo(), columns=['date','mean'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    fig, ax = plt.subplots(figsize=(15,10))
                                        # we'll create the plot by setting our dataframe to the data argument
                    sns.lineplot(data=df, ax=ax)

                                        # we'll set the labels and title
                    ax.set_ylabel('PLC # Y50',fontsize=20)
                    ax.set_xlabel('Time',fontsize=20)
                    ax.set_title('Daily Mean of Western Hemlock  Percent Loss of Conductance # EXTREME DRY SOIL CONDITION',fontsize=20);
                    ax.grid()
                    output_widget = Output()
                    
                    with output:
                        output.clear_output()
                        plt.show() 
                    

        
            if change['new'] == 'Reset':
                output.clear_output()
                mdwp_df_condition.value = None
                mdwp_wh_condition.value = None
                climate.value = None
                plc_df.value = None
                plc_wh.value = None
                database.value = 'Real-Time Dataset (GRIDMET)'
                scenario.value = None
                model.value = None
    btns.observe(btns_button_click, 'value')



    # In[11]:


    from ipywidgets import Button, GridBox, Layout, ButtonStyle


    # In[12]:


    from ipywidgets import AppLayout, Button, Layout
    header_button = create_expanded_button('Header', 'success')
    left_button = create_expanded_button('Left', 'info')
    center_button = create_expanded_button('Center', 'warning')
    right_button = create_expanded_button('Right', 'info')
    footer_button = create_expanded_button('Footer', 'success')


    # In[13]:


    AppLayout(header=None,
              left_sidebar=output,
              center=Map,
              right_sidebar=None,
              footer=None)


    # In[14]:


    header  = Button(description='Header',
                     layout=Layout(width='auto', grid_area='header'),
                     style=ButtonStyle(button_color='lightblue'))
    main    = Button(description='Main',
                     layout=Layout(width='auto', grid_area='main'),
                     style=ButtonStyle(button_color='turquoise'))
    sidebar = Button(description='Sidebar',
                     layout=Layout(width='auto', grid_area='sidebar'),
                     style=ButtonStyle(button_color='salmon'))
    footer  = Button(description='Footer',
                     layout=Layout(width='auto', grid_area='footer'),
                     style=ButtonStyle(button_color='moccasin'))

    GridBox(children=[header, main, footer],
            layout=Layout(
                width='100%',
                grid_template_rows='auto auto auto auto',
                grid_template_columns='25% 25% 25% 25%',
                grid_template_areas='''
                "header header header header"
                "main main main main "
                "footer footer footer footer"
                ''')
           )


    # In[15]:


    header.description = 'Contact: carlos.gonzalez@oregonstate.edu & karabass@oregonstate.edu'
    main.description = 'Developed by Sefa Karabas'
    footer.description = 'All rights reserved @ 2022'


    # In[16]:


    from IPython.display import Image
    from IPython.core.display import HTML 
    Image(url= "https://communications.oregonstate.edu/sites/communications.oregonstate.edu/files/osu-primarylogo-2-compressor.jpg",width=300,height=300)

