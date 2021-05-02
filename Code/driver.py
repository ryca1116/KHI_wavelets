import netCDF4 as nc 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import Packages.waveletFunctions as wv
import os
#import pywt as wv

class windData:
    def __init__(self, filename = None, height = None):
        self.filename = filename
        self.date     = self.get_date()
        self.datObj   = nc.Dataset(self.filename)
        self.varNames = self.get_varNames()
        self.dat      = self.get_data()
        self.get_velocities()
        self.times     = self.parse_dat('time')
        self.heights  = self.parse_dat('height')

        # Set the wave data for the object
        if height == None: # If no height is specified, return all wave data as 3D matrix
            for iHeight in np.arange(len(list(self.heights))): # Loop through each height
                waveDict = self.wv_analyze('w',iHeight)
                wave     = waveDict['wave']
                if iHeight == 0: # First loop, set the variable size
                    waveShape = wave.shape
                    fullWave = np.zeros(waveShape + (len(list(self.heights)),))
                    fullWave[:,:,iHeight] = wave
                else: # Append data to the end of the matrix
                    fullWave[:,:,iHeight] = wave
            self.wave = fullWave
            dataShape = self.u.shape
        else: # If the height value for the object is specified
            heights     = np.array(list(self.dat['height'][:]))
            overHeights = heights[heights >= height]
            searchTerm  = overHeights[0]
            index       = np.where(heights == searchTerm)
            iHeight      = index[0][0]

            waveDict = self.wv_analyze('w',iHeight)
            wave     = waveDict['wave']
            self.wave = wave
            self.scales = waveDict['scale']
            self.sig = waveDict['sig']

            dataShape = self.u.shape
            self.u = self.u[:,iHeight]
            self.v = self.v[:,iHeight]
            self.w = self.w[:,iHeight]

        # Set the shear data for the object
        timeShear = np.zeros(dataShape)
        for i in np.arange(len(list(self.times))):
            # For each timestep get the shear 
            shears = self.shear_dat(i)
            fullShear = shears['fullDer']
            fullShear = np.transpose(fullShear) # Put it as a column vector
            timeShear[i,:] = fullShear # Add it to the total matrix
        timeShear = np.transpose(timeShear)
        self.shear = timeShear[0:20,:]
        
    def parse_dat(self,dataName):
        # PARSE_DATA returbs a numpy array of the data requested in dataName
        dat = self.dat[dataName][:]
        dat = dat.data

        return dat

    def get_date(self):
        # GET_DATE gets the date in mm-dd-yyyy format from the filename
        namesplit = self.filename.split('.')
        dateStr   = namesplit[2]
        year  = str(dateStr[0:4])
        month = str(dateStr[4:6])
        day   = str(dateStr[6:8])

        fullDate = '-'.join([month,day,year])
        return fullDate

    def get_varNames(self):
        # This function gets the varaible names and assigns it to the object property
        vars = self.datObj.variables
        names = list(vars.keys())
        return names

    def get_data(self):
        # This function retrieves the data from the filename and stores it all in a dictionary. Returns nothinf but sets the property of the class object 'dat' to the data dictionary
        dataObject = self.datObj
        # Get all the variable names
        dataDict = {}
        for i in self.varNames: # Loop through the varaible names and assign them
            varData = dataObject[i]
            dataDict[i] = varData
        
        # Now assign the object property
        return dataDict

    def get_velocities(self,velVar = None):
        # GET_VELOCITIES gets all three velocity components and returns them in a dictionary
        if velVar != None: # Getting a specific dataSet
            varData = self.dat[velVar][:]
            varData = varData.data
            
            return varData
        else: # General method that sets the variables
            wDat = self.dat['w'][:]
            wDat = wDat.data

            uDat = self.dat['u'][:]
            uDat = uDat.data

            vDat = self.dat['v'][:]
            vDat = vDat.data

            self.u = uDat
            self.w = wDat
            self.v = vDat

    def wv_analyze(self, velocityVar, heightInd):
        # This function performs wavelet analysis on the desired velocity set in time and height. 
        # Find time increment
        time = self.dat['time'][:]
        time = list(time.data)
        dt = time[1]-time[0]
        index = heightInd
        # Get the time series data and analyze
        velData = self.get_velocities(velVar=velocityVar)
        wvDat = list(velData[:,index])
        wave, period, scale, coi = wv.wavelet(wvDat,dt,pad=1,dj=0.1)

        # Do significance testing
        #sig,noise = wv.wave_signif(wvDat,dt,scale)
        sig = None

        return {'wave':wave, 'period':period, 'scale': scale, 'coi':coi, 'waveDat':wvDat, 'time':time, 'velDat':velData, 'sig':sig}

    def shear_dat(self, timeInd):
        # SHEAR_DAT returns the shear data as a time series to compare with the frequency and 
        # Start with velocity data
        uDat = self.dat['u'][:]
        uDat = uDat[timeInd,:]
        uDat = list(uDat.data)
        
        vDat = self.dat['v'][:]
        vDat = vDat[timeInd,:]
        vDat = list(vDat.data)

        height = self.dat['height'][:]
        height = list(height.data)

        # Do the numerical differentiation
        uDer = self.num_diff(uDat,height)
        vDer = self.num_diff(vDat,height)
        fullDer = list(np.sqrt(np.multiply(uDer,uDer) + np.multiply(vDer,vDer)))

        return {'uDer': uDer, 'vDer':vDer, 'fullDer':fullDer, 'height':height}

    def num_diff(self, velocity, time):
        # NUM_DIFF computes the numerical derivative of the time series data using the central difference method
        der = np.zeros(len(velocity))
        for i in np.arange(len(velocity)): # Loop through each index of the timeSeries
            if i == 0: # First index use forward difference
                h = time[i+1] - time[i]
                der[i] = (velocity[i + 1] - velocity[i])/h
            elif i == (len(velocity)-1): # Backward difference method
                der[i] = (velocity[i] - velocity[i-1])/h
            else: # Central difference 
                h = time[i+1] - time[i]
                der[i] = (velocity[i+1] - velocity[i-1])/(2*h)
            
        return der                   

def get_data_files(dirName):
    # GET_DATA_FILES is a function that gets all the filenames of the data files in a specified directory
    dirlist = list(os.walk(dirName))
    dirTup  = dirlist[0] # This is a tuple, the third value is the filenames
    filenames = dirTup[2]

    return filenames

def sliding_av(time,dataSeries):
    # SLIDING_AV computes a windowed average of time series data and returns the modified time vector as well as the averaged data
    windowSize = 5 # Size of the window average to compute
    seriesData = pd.Series(dataSeries)
    windows = seriesData.rolling(windowSize)
    dataAv = windows.mean()
    dataAv = np.array(list(dataAv))
    # Get rid of the NaN's
    dataAv = dataAv[windowSize-1:]
    time = time[windowSize-1:]

    return time,dataAv

def plots(dataObj,save=0,show=0):
    # PLOTS plots all of the necessary data in an organized and intuitive fashion
    ####################################################################################################################
    # Plot the heatmaps
    waveDat = dataObj.wave
    wave = np.real(waveDat)

    # Now plot
    plt.close('all') # Make sure no other figures are open
    # Plot the velocity components
    fig, axs = plt.subplots(3,1)
    fig.set_size_inches(18,12)
    fig.suptitle(dataObj.date)
    axs[0].plot(dataObj.times, dataObj.w,label='Raw Data')
    avTime,avW = sliding_av(dataObj.times,dataObj.w)
    axs[0].plot(avTime,avW,label='Averaged')
    axs[0].set_title("W Velocity")
    endTime = dataObj.times[-1]
    interval = endTime/4
    axs[0].set_xticks([1,interval,interval*2,interval*3,interval*4])
    axs[0].set_xticklabels(labels=["12:00AM","6:00AM","12:00PM","6:00PM","12:00PM"],rotation='horizontal')
    axs[0].set_ylabel('Velocity')
    axs[0].legend()

    hVel = np.sqrt(np.multiply(dataObj.u,dataObj.u) + np.multiply(dataObj.v,dataObj.v))
    axs[1].plot(dataObj.times,hVel,label='Raw data')
    avTime,avW = sliding_av(dataObj.times,hVel)
    axs[1].plot(avTime,avW,label='Averaged')
    axs[1].set_title("Horiztal Velocity")
    axs[1].set_xticks(ticks=[1,interval,interval*2,interval*3,interval*4])
    axs[1].set_xticklabels(["12:00AM","6:00AM","12:00PM","6:00PM","12:00PM"],rotation='horizontal')
    axs[0].set_ylabel('Velocity')
    axs[1].legend()
    ####################################################################################################################
    # Plot the shear data as a heatmap
    waveFig, waveAxs = plt.subplots(2,1)
    waveFig.set_size_inches((12,8))
    shearDat = dataObj.shear
    shearInt = int(shearDat.shape[1]/4)
    shearSlice = shearDat[:,shearInt*3-1:shearInt*4-1]
    htMap = sns.heatmap(shearSlice,ax=waveAxs[0])
    # Set the tickmarks in the time domain
    time = list(dataObj.times)
    xTickMarks = [1,shearSlice.shape[1]]
    xTickLabels = ["6:00PM","12:00PM"]
    #xTickMarks = [1,interval,interval*2,interval*3,interval*4]
    #xTickLabels = ["12:00AM","6:00AM","12:00PM","6:00PM","12:00PM"]
    htMap.set_xticks(ticks=xTickMarks)
    htMap.set_xticklabels(xTickLabels,rotation='horizontal')
    htMap.set_title("Wind Shear on " + dataObj.date)
    # Set the yticks for the heights, remember do this backwards
    heights = dataObj.heights
    heights = list(heights[0:20].data)
    tickSpace = 50
    upLim  = np.floor((heights[-1])/tickSpace)*tickSpace
    botLim = np.ceil((heights[0])/tickSpace)*tickSpace
    numels = ((upLim-botLim)/tickSpace) + 1
    perTicks = np.linspace(botLim,upLim,int(numels)) # Ticks in the period
    perTicks = list(map(float,perTicks))
    # Relate the scale numbers to the tick index
    scalesLog = list(heights)
    m         = scalesLog[1]-scalesLog[0]
    b         = scalesLog[0]
    inds      = list(np.divide(np.array(perTicks) - b, m))
        
    for i in np.arange(len(perTicks)):
        perTicks[i] = int(perTicks[i])
        
    perTicks = list(map(str,perTicks))
    perTicks[-1] = ""
    htMap.set_yticks(inds)
    htMap.set_yticklabels(perTicks)
    htMap.axes.xaxis.set_tick_params(pad=10)
    # Flip the y axis so its in creasing height order
    newLims = htMap.axes.get_ylim()[::-1]
    htMap.axes.set_ylim((0,newLims[-1]))
    # Add Axis labels
    htMap.set_ylabel('Height [m]')

    # Plot the wavelet spectrogram
    waveMap = sns.heatmap(wave,vmax = 1.0,vmin=-1.0, ax=waveAxs[1])#cmap='RdYlGn'
    waveMap.set_xlabel('Time at location')
    waveMap.set_ylabel('Period [min]')
    # Set the time ticks
    waveMap.set_xticks(ticks=[1,24,48,72,96])
    waveMap.set_xticklabels(labels=["12:00AM","6:00AM","12:00PM","6:00PM","12:00PM"],rotation='horizontal')
    # Set the yticks as frequencies
    lamb    = 1.03 # From Torrence et. Compo for MORLET wavelets
    scales  = dataObj.scales
    periods = dataObj.scales*lamb # in s
    periods = periods/(60) # in minutes
    freq    = 1/periods # This is what we want to plot in
    freq = freq*(1e6)
    tickSpace = 200
    upLim = np.floor((periods[-1])/tickSpace)*tickSpace
    numels = (upLim/tickSpace) + 1
    perTicks = np.linspace(0,upLim,int(numels)) # Ticks in the period
    perTicks = list(map(float,perTicks))
    scaleTicks = list(np.multiply(perTicks,(60/lamb))) # Scale numbers the ticks occur at
    # Relate the scale numbers to the tick index
    scalesLog = list(np.log(scales))
    m         = scalesLog[1]-scalesLog[0]
    b         = scalesLog[0]
    inds      = list(np.divide(list(np.log(scaleTicks)) - b, m))
    inds[0]   = 0.0
    waveMap.set_yticks(inds)
    waveMap.set_yticklabels(list(map(str,perTicks)))
    waveMap.set_title("Spectrogram")
    ####################################################################################################################
    # Now get the max power spectrum and the frequency it occurs at
    maxTup = np.where(wave == np.amax(wave))
    freqVec = np.zeros(wave.shape[1])
    for i in np.arange(wave.shape[1]): # Get max frequency for each time
        timeWave = wave[:,i]
        ind = np.where(timeWave == np.amax(timeWave))
        timeScale = scales[ind]
        freq = 1/(lamb*timeScale)
        freqVec[i] = freq
    
    scaleInd = maxTup[0][0]
    col = maxTup[1][0]
    maxScale = scales[scaleInd]
    maxFreq = 1/(lamb*maxScale)
    times = list(dataObj.times)
    maxTime = times[col]
    ####################################################################################################################
    # Calculate the estimated richardson number and plot it
    N = freqVec # Say this is the Brunt-Vaisla frequency
    hubShear = shearDat[0,:] # Just the hub height shear
    Ri = np.divide(np.multiply(N,N),np.multiply(hubShear,hubShear))
    axs[2].plot(dataObj.times,Ri)
    axs[2].set_xticks([1,interval,interval*2,interval*3,interval*4])
    axs[2].set_xticklabels(labels=["12:00AM","6:00AM","12:00PM","6:00PM","12:00PM"],rotation='horizontal')
    axs[2].set_title("Richardson Number Evolution")
    axs[2].set_ylabel('Ri')
    axs[2].set_xlabel('Time at Location')
    ####################################################################################################################
    # Plot shear data over time for the hub height
    shFig,shAx = plt.subplots(1,1)
    shFig.set_size_inches(12,8)
    timeShear = shearDat[0,:]
    avTime,avShear = sliding_av(dataObj.times,timeShear)
    shAx.plot(dataObj.times,timeShear,label='Raw data')
    shAx.plot(avTime,avShear,label='Average')
    shAx.set_xticks([1,interval,interval*2,interval*3,interval*4])
    shAx.set_xticklabels(labels=["12:00AM","6:00AM","12:00PM","6:00PM","12:00PM"],rotation='horizontal')
    shAx.set_ylabel('Shear Data')
    shAx.set_xlabel('Time at Location')
    shAx.set_title("Wind Shear at Hub Height " + dataObj.date)
    shAx.legend()
    # Plot the significace data as a result of the chi-squared test
    #sigDat = dataObj.sig
    #sigFig,sigAx = plt.subplot(1,1)
    #sigAx.plot(dataObj.times,sigDat)
    #sigAx.set_xticks([1,interval,interval*2,interval*3,interval*4])
    #sigAx.set_xticklabels(labels=["12:00AM","6:00AM","12:00PM","6:00PM","12:00PM"],rotation='horizontal')
    #sigAx.set_title("Chi squared significance " + dataObj.date)
    #sigAx.set_ylabel('Significance')
    #sigAx.set_xlabel('Time at Location')
    if show == 1:
        plt.show()
    elif save == 1:
        splitName = dataObj.filename.split('.')
        joinedName = '_'.join(splitName[0:len(splitName)-1])
        saveName = joinedName.split('/')
        saveName[1] = 'Figs'
        saveName[2] = dataObj.date
        saveName = '/'.join(saveName)
        waveName = saveName + "_wave.jpg"
        velName  = saveName + "_vels.jpg"
        #sigName   = saveName + "_sig.jpg"
        shearName = saveName + "_shear.jpg"
        fig.savefig(velName)
        waveFig.savefig(waveName)
        shFig.savefig(shearName)
        #sigFig.savefig(sigName)

    return {'velocityFig': fig, 'waveFig': waveFig, 'maxFreq':maxFreq, 'maxTime':maxTime, 'timeInd':col, 'ri':Ri[col]}


# Get the filenames for the data
pathName = "Code/Data"
filenames = get_data_files(pathName)
numFile = len(filenames)
varNames = ["date","maxFreq","maxTime"]
dateVec = []
freqVec = []
timeVec = []
periVec = []
riVec   = []
for fileName in filenames: # Loop through each file
    # Test out the variables
    fullName = pathName + "/" + str(fileName)
    # Set object with wave, shear, velocity, time, and height data
    dataObj = windData(filename = fullName, height=80)
    # Plot everything
    figData = plots(dataObj,save=1)
    # Create pandas dataframe of the frequency and time data
    dateVec = dateVec + [dataObj.date]
    freqVec = freqVec + [figData['maxFreq']]
    periVec = periVec + [(1/figData['maxFreq'])/60]
    timeVec = timeVec + [figData['maxTime']]
    riVec   = riVec   + [figData['ri']]
    # Calculate the estimated Richardson number

csvDict  = {"maxFreq": freqVec, "period":periVec, "maxTime":timeVec, "Ri":riVec}
csvFrame = pd.DataFrame(data=csvDict,index=dateVec)
csvFrame.to_csv('Code/Outputs/frequency.csv')