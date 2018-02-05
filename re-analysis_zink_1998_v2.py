import os
import copy
import pickle

import numpy
import matplotlib
matplotlib.use('Agg')
font = {'family':'Ubuntu'}
matplotlib.rc('font', **font)
from matplotlib import pyplot


# # # # #
# SETTINGS

# Recompute all the fits from the Zink data, or load from pickled files.
RECOMPUTE = True

# FITTING
# Several models are fitted to the data from Zink et al. by means of a full
# exploration of parameter space. This is slow compared to more intelligent
# fitting procedures that home in on the best fitting set of parameters by
# identifying local minima/maxima. However, this is the most thorough way of
# characterising parameter space.
# Ranges for the a and b parameters.
ARANGE = { \
    'lin':      [0, 3.5], \
    'lin-prop': [0, 3.5], \
    'exp':      [0, 7], \
    }
BRANGE = { \
    'lin':      [0, 3.5], \
    'lin-prop': [0, 3.5], \
    'exp':      [0, 7], \
    }
# Stepsize for the a and b parameters.
ASTEP = { \
    'lin':      0.0005, \
    'lin-prop': 0.0005, \
    'exp':      0.001, \
    }
BSTEP = { \
    'lin':      0.0005, \
    'lin-prop': 0.0005, \
    'exp':      0.001, \
    }

# PLOTTING
# Plot dimensions in inches.
FIGSIZE = (8.0, 6.0)
# Dots per inch in the plot.
FIGDPI = 100.0
# File extension of the plots, e.g. 'png' or 'svg'.
FIGEXT = 'png'
# Set limits to the GVS axis (in mA).
GVSAXLIM = [0, 8]
# Set limits to the tilt/torsion axis.
TAXLIM = [0, 6]
# Line width for averages.
LINEWIDTH = 3
# Opacity value for the standard deviation range.
SDALPHA = 0.3
# Opacity value for the min/max values.
MALPHA = 0.1
# Define the colours for plots.
PLOTCOLS = { \
    'tilt':     '#c4a000', \
    'torsion':  '#204A87', \
    'combined': '#4e9a06', \
    'error':    '#2e3436', \
    }
# Define the font sizes for each type of graph element.
FONTSIZE = {\
    'axis':     18, \
    'legend':   12, \
    }
# The absolute maximim squared residuals value that we'll use for plotting.
VABSMAX = 5.0
# Set the tick labels for the parameter space.
XTICKS = { \
    'lin':      [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], \
    'lin-prop': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], \
    'exp':      [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], \
    }
YTICKS = { \
    'lin':      [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], \
    'lin-prop': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], \
    'exp':      [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], \
    }

# FILES AND FOLDERS
# Auto-detect the name of this file.
this_file = os.path.basename(os.path.abspath(__file__))
this_file_name, this_file_ext = os.path.splitext(this_file)
# Auto-detect the path to the directory that this file is in.
DIR = os.path.dirname(os.path.abspath(__file__))
# Generate the path to the output directory.
OUTDIR = os.path.join(DIR, 'output_%s' % (this_file_name))
# Check if the output directory exists, and create a new one if it doesn't
# exist yet.
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)


# # # # #
# DATA

# Directly copied from Zink et al. (1998) manuscript, where it is listed in
# tables. Slighlty awkward, but it works.
zink_data = { \
    'tilt': { \
        'gvs_intensity':    numpy.array([1.5, 2.0, 2.5, 3.0]), \
        'N':                numpy.array([4, 12, 12, 4]), \
        'left': { \
            'm':    numpy.array([2.2, 2.6, 3.2, 4.9]), \
            'sd':   numpy.array([0.9, 1.4, 2.3, 1.5]), \
            'min':  numpy.array([1.3, 1.3, 1.2, 3.0]), \
            'max':  numpy.array([3.3, 6.3, 9.4, 6.5]), \
            }, \
        'right': { \
            'm':    numpy.array([1.7, 2.6, 3.1, 4.8]), \
            'sd':   numpy.array([0.5, 1.2, 2.0, 1.8]), \
            'min':  numpy.array([1.3, 1.0, 1.0, 2.6]), \
            'max':  numpy.array([2.3, 5.8, 8.5, 6.4]), \
            }, \
        }, \

    'torsion': { \
        'gvs_intensity':    numpy.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), \
        'N':                numpy.array([6, 2, 7, 7, 6, 3, 3, 2]), \
        'left': { \
            'm':    numpy.array([1.0, 1.3, 2.0, 2.5, 2.9, 3.2, 3.6, 3.9]), \
            'sd':   numpy.array([0.4, 0.1, 0.5, 0.8, 1.0, 1.1, 1.3, 1.8]), \
            'min':  numpy.array([0.5, 1.2, 1.3, 1.4, 1.2, 2.0, 2.2, 2.6]), \
            'max':  numpy.array([1.5, 1.4, 2.5, 3.5, 3.8, 4.1, 4.5, 5.2]), \
            }, \
        'right': { \
            'm':    numpy.array([1.2, 1.4, 2.1, 3.0, 3.3, 3.6, 4.1, 4.0]), \
            'sd':   numpy.array([0.3, 0.1, 0.5, 0.6, 1.3, 1.9, 1.7, 2.1]), \
            'min':  numpy.array([0.6, 1.3, 1.5, 2.2, 1.3, 1.5, 2.2, 2.5]), \
            'max':  numpy.array([1.4, 1.4, 2.8, 3.5, 4.2, 4.3, 5.2, 5.4]), \
            }, \
        }, \
    }


# # # # #
# AVERAGE

# Zink et al. did left and right stimulation, so we can average their efforts.
measures = zink_data.keys()
measures.sort()
for m in measures:
    # Create a new key in the zink_data dict.
    zink_data[m]['avg'] = {}
    # Average the means and standard deviations.
    zink_data[m]['avg']['m'] = \
        (zink_data[m]['left']['m'] + zink_data[m]['right']['m']) / 2.0
    zink_data[m]['avg']['sd'] = \
        (zink_data[m]['left']['sd'] + zink_data[m]['right']['sd']) / 2.0
    # Find the minimum and maximum values.
    zink_data[m]['avg']['min'] = \
        numpy.min(numpy.vstack((zink_data[m]['left']['min'], \
        zink_data[m]['right']['min'])), axis=0)
    zink_data[m]['avg']['max'] = \
        numpy.max(numpy.vstack((zink_data[m]['left']['max'], \
        zink_data[m]['right']['max'])), axis=0)


# # # # #
# MODEL FITTING

def vest_func(x, a, b, model='lin', measure='tilt'):
    
    """Returns the results of a function applied to the passed x values.
    
    Arguments
    
    x        -  NumPy array with shape (M,). This represents the vestibular
                input.

    a        -  float of the a parameter value. This value determines the
                slope in all of the equations.
                
    b       -   float of the b parameter value. This is ignored if model=='lin',
                determines slope if model=='exp', and is the exponent when
                model=='pow'.
    
    Keyword Arguments
    
    model   -   string that indicates the specific function. Can be: 'lin' for
                a linear model y=a*x+b, 'lin-prop' for a directly proportional
                linear function y=a*x, 'exp' for a function that exponentially
                increases or decreases in slope y = a*(1-exp(-x/b)) or
                y = a*(exp(x/b)-1).
    
    measure -   string that indicates what measure is to be fitted. Can be
                'torsion' to fit an exponentially decreasing function, or
                'tilt' to fit an exponentially increasing function when
                model='exp'. Linear models will always be fitted in the same
                way regardless of measure.
    
    Returns
    
    y       -   The result of the function described in 'model' (given what
                measure is defined).
    """
    
    # Directly proportional function with slope a and intercept 0.
    if model == 'lin-prop':
        return a * x
    # Linear function with slope a and intercept b.
    elif model == 'lin':
        return a * x + b
    # Exponetially decreasing function that passes through origin.
    # (a determines slope, b determines asymptote).
    elif model == 'exp' and measure == 'torsion':
        #return b + -b / (a**x)
        return b * (1 - numpy.exp(-x / a))
    # Exponentially increasing function
    # (a determines slope, b determines exponential increase)
    elif model == 'exp' and measure == 'tilt':
        #return a * (x**b)
        return b * (numpy.exp(x / a) - 1)


# Load or recompute the parameters.
if not RECOMPUTE:
    print("Loading parameter space and estimates from pickle files.")
    with open('param.pickle', 'rb') as f:
        param = pickle.load(f)
    with open('paramspace.pickle', 'rb') as f:
        paramspace = pickle.load(f)

else:
    # Loop through all measures we want to model.
    param = {}
    paramspace = {}
    for m in ['tilt', 'torsion']:
        param[m] = {}
        paramspace[m] = {}
        for model in ['lin-prop', 'lin', 'exp']:
            # Start with NaN parameters and R=0.
            param[m][model] = {'a':numpy.nan, 'b':numpy.nan, 'R':0}
            paramspace[m][model] = numpy.zeros( \
                (int((BRANGE[model][1]-BRANGE[model][0])/BSTEP[model])+1, \
                int((ARANGE[model][1]-ARANGE[model][0])/ASTEP[model])+1), \
                dtype=float)
            # Get the data from Zink et al.
            x = zink_data[m]['gvs_intensity']
            y = zink_data[m]['avg']['m']
            # Compute the total sum of squares.
            SStot = numpy.sum((y - numpy.mean(y))**2)
            # Report on what's currently happening.
            print("Fitting '%s' model to %s data." % (model, m))
    
            # Loop through a values.
            arange = numpy.arange(ARANGE[model][0], \
                ARANGE[model][1]+ASTEP[model], ASTEP[model])
            for i, a in enumerate(arange):
                # Loop through b values.
                brange = numpy.arange(BRANGE[model][0], \
                    BRANGE[model][1]+BSTEP[model], BSTEP[model])
                if model == 'lin-prop':
                    brange = [0.0]
                for j, b in enumerate(brange):
    
                    # Calculate the predicted values.
                    y_pred = vest_func(x, a, b, model=model, measure=m)
                    # Compute the residuals.
                    res = y - y_pred
                    # Compute the sum of squares of the regression.
                    SSreg = numpy.sum((y_pred - numpy.mean(y))**2)
                    # Compute the residual sum of squares.
                    SSres = numpy.sum(res**2)
                    # Only when the sum of squares of the total is roughly equal
                    # to SSreg + SSres, in which case the R square = explained
                    # variance. NOTE: This number could be 0.1 for accurate models,
                    # but needs to be a bit higher, e.g. 10, for the directly
                    # propotional linear models, as they fit quite poorly.
                    if abs(SStot - (SSreg+SSres)) < 10:
                        # Compute the the coefficient of determination (R squared).
                        r_sq = 1 - (SSres / SStot)
                        # Compute R. It's the square root of R squared.
                        r = numpy.sqrt(r_sq)
                        # If r is the best estimate up to now, save the parameters.
                        if r > param[m][model]['R']:
                            param[m][model] = { \
                                'a':copy.deepcopy(a), \
                                'b':copy.deepcopy(b), \
                                'R':copy.deepcopy(r), \
                                }
    
                    # Save the current residual sum of squares.
                    if model == 'lin-prop':
                        paramspace[m][model][:,i] = SSres #r_sq
                    else:
                        paramspace[m][model][j,i] = SSres #r_sq

# Save the parameter estimates.
print("Writing parameter estimates to a text file.")
with open(os.path.join(OUTDIR, 'parameter_estimates.csv'), 'w') as f:
    header = ['measure', 'model', 'a', 'b', 'R']
    f.write(','.join(map(str, header)))
    for m in param.keys():
        for model in param[m].keys():
            line = [m, model, param[m][model]['a'], param[m][model]['b'], \
                param[m][model]['R']]
            f.write('\n' + ','.join(map(str, line)))

# Save some computed stuff.
if RECOMPUTE:
    print("Storing parameter space in and estimates in pickle files.")
    with open('param.pickle', 'wb') as f:
        pickle.dump(param, f)
    with open('paramspace.pickle', 'wb') as f:
        pickle.dump(paramspace, f)


# # # # #
# PLOTS

# Plot the parameter spaces.
print("Plotting parameter spaces.")
for figtype in ['bigfig', 'sepfigs']:
    # Create a big figure if needed.
    if figtype == 'bigfig':
        bigfig, bigax = pyplot.subplots(nrows=3, ncols=2, \
            figsize=(FIGSIZE[0]*2, FIGSIZE[1]*3), dpi=FIGDPI)

    # Loop through all measures and models.
    for j, m in enumerate(['torsion', 'tilt']):
        for i, model in enumerate(['lin', 'lin-prop', 'exp']):

            # Choose whether to create a new figure or to choose one of the
            # axes from the big figure.
            if figtype == 'sepfigs':
                fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=FIGSIZE, \
                    dpi=FIGDPI)
            else:
                ax = bigax[i,j]

            # Copy the parameter space.
            plotspace = numpy.copy(paramspace[m][model])
            # Find all the NaNs and all the infinites.
            notnan = numpy.isnan(plotspace) == False
            notinf = numpy.isinf(plotspace) == False
            # Find the maximum value.
            maxval = numpy.max(plotspace[notnan & notinf])
            # Replace 'inf' by the highest number.
            plotspace[notinf==False] = maxval
            # Replace 'nan' by the highest number.
            plotspace[notnan==False] = maxval
            # Find the minimum value, and compute the value range.
            minval = numpy.min(plotspace[notnan & notinf])
            vrange = maxval - minval
            # Replace values higher than the maximum by the maximum, to increase
            # the resolution of the lower values (the lower values are more densly
            # packed and more interesting, so we want to see those in the plot).
            plotspace[plotspace>VABSMAX] = VABSMAX
            # Get the space of the plot for the plot labels.
            cxlbl = numpy.arange(0, VABSMAX+0.01, (VABSMAX)/10.0)
            # Currently, low values reflect better fits, so we need to reverse
            # the space.
            plotspace = numpy.max(plotspace) - plotspace
            cxlbl = numpy.max(cxlbl) - cxlbl
            # Plot the values across parameter space.
            cax = ax.imshow(plotspace, vmin=0.0, vmax=VABSMAX, \
                cmap='inferno', aspect='equal', interpolation='none')
            # Plot the best fitting parameter.
            ax.scatter(param[m][model]['a']/ASTEP[model], \
                param[m][model]['b']/BSTEP[model], \
                s=100, c=(0,0,0,0), marker='o', linewidths=2, \
                edgecolors='#ff69b4')
            # Plot a colour bar.
            if figtype == 'sepfigs':
                cx = numpy.arange(0.0, VABSMAX+0.01, (VABSMAX)/10.0)
                cbar = fig.colorbar(cax, ticks=cx, orientation='vertical')
                cbar.ax.set_ylabel("Residual sum of squares")
                cbar.ax.set_yticklabels(numpy.round(cxlbl, decimals=2))
            # Set the x-axis limits and ticks.
            ax.set_xlim([0, paramspace[m][model].shape[1]-1])
            xticks = numpy.arange(0, paramspace[m][model].shape[1]+1, \
                paramspace[m][model].shape[1] / (len(XTICKS[model])-1))
            ax.set_xticks(xticks)
            ax.set_xticklabels(XTICKS[model])
            # Set the y-axis limits and ticks.
            ax.set_ylim([0, paramspace[m][model].shape[0]-1])
            yticks = numpy.arange(0, paramspace[m][model].shape[0]+1, \
                paramspace[m][model].shape[0] / (len(YTICKS[model])-1))
            ax.set_yticks(yticks)
            ax.set_yticklabels(YTICKS[model])
            # Set the axis labels.
            if (figtype == 'sepfigs') or (figtype=='bigfig' and model=='exp'):
                ax.set_xlabel("$a$ parameter", fontsize=FONTSIZE['axis'])
            if (figtype == 'sepfigs'):
                ax.set_ylabel("$b$ parameter", fontsize=FONTSIZE['axis'])
            elif figtype=='bigfig' and m=='torsion':
                ax.set_ylabel("%s\n\n$b$ parameter" % (model), \
                    fontsize=FONTSIZE['axis'])
            # Set the figure title.
            if figtype == 'bigfig' and model == 'lin':
                ax.set_title("%s" % (m.title()))
            # Save the plot if this is a separate figure.
            if figtype == 'sepfigs':
                fig.savefig(os.path.join(OUTDIR, 'parameter_space_%s_%s.%s' % \
                    (m, model, FIGEXT)))
                pyplot.close(fig)
    # Save the big figure.
    if figtype == 'bigfig':
        bigfig.savefig(os.path.join(OUTDIR, 'parameter_space_full.%s' % (FIGEXT)))
        pyplot.close(bigfig)


# Average plot, showing average tilt and torsion as a function of GVS intensity.
print("Plotting torsion and tilt as a function of stimulation intensity.")
fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=FIGSIZE, dpi=FIGDPI)
# Loop through the data.
measures = zink_data.keys()
measures.sort()
for m in measures:
    # Get the x-values, which in this case are the stimulation intensities.
    x = zink_data[m]['gvs_intensity']
    # Get the y-values, which in this case are average tilt or torsion.
    y = zink_data[m]['avg']['m']
    # Get the shaded area, the standard error of the mean.
    y1 = y - zink_data[m]['avg']['sd'] / numpy.sqrt(zink_data[m]['N'])
    y2 = y + zink_data[m]['avg']['sd'] / numpy.sqrt(zink_data[m]['N'])
    # Plot the average tilt/torsion.
    ax.plot(x, y, '-', lw=LINEWIDTH, color=PLOTCOLS[m], label='%s' % (m.title()))
    # Plot the shaded areas.
    ax.fill_between(x, y1, y2, color=PLOTCOLS[m], alpha=SDALPHA)
    # Plot the modelled data accoding to the exponential model.
    x_ = numpy.arange(GVSAXLIM[0], GVSAXLIM[1]+0.01, 0.01)
    y_pred = vest_func(x_, param[m]['exp']['a'], param[m]['exp']['b'], \
        model='exp', measure=m)
    ax.plot(x_, y_pred, '--', lw=2, color=PLOTCOLS[m], \
        label=r'Fit ($R^2$=%.2f)' % (param[m]['exp']['R']**2))
    # Plot the modelled data accoding to the linear model.
    y_pred = vest_func(x_, param[m]['lin-prop']['a'], \
        param[m]['lin-prop']['b'], model='lin-prop', measure=m)
    ax.plot(x_, y_pred, ':', lw=2, color=PLOTCOLS[m], \
        label=r'Fit ($R^2$=%.2f)' % (param[m]['lin-prop']['R']**2))
# Draw the legend.
ax.legend(loc='lower right', fontsize=FONTSIZE['legend'])
# Set the axis limits.
ax.set_xlim(GVSAXLIM)
ax.set_ylim(TAXLIM)
# Write the axis labels.
ax.set_xlabel("Galvanic vestibular stimulation (mA)", fontsize=FONTSIZE['axis'])
ax.set_ylabel(r"Ocular torsion or visual tilt ($\degree$)", \
    fontsize=FONTSIZE['axis'])
# Save the plot.
fig.savefig(os.path.join(OUTDIR, 'torsion_and_tilt.%s' % (FIGEXT)))
# Close the plot.
pyplot.close(fig)


# Plot of visual tilt as a function of ocular torsion under galvanic vestibular
# stimulation.
print("Plotting combined plot (visual tilt as a function of ocular torsion).")
fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=FIGSIZE, dpi=FIGDPI)
# Find the GVS intensities used in both.
shared_intensities = numpy.intersect1d(zink_data['torsion']['gvs_intensity'], \
    zink_data['tilt']['gvs_intensity'])
xsel = numpy.in1d(zink_data['torsion']['gvs_intensity'], shared_intensities)
ysel = numpy.in1d(zink_data['tilt']['gvs_intensity'], shared_intensities)
# Find the datapoints to plot.
x = zink_data['torsion']['avg']['m'][xsel]
xsem = zink_data['torsion']['avg']['sd'][xsel] \
    / numpy.sqrt(zink_data['torsion']['N'][xsel])
y = zink_data['tilt']['avg']['m'][ysel]
ysem = zink_data['tilt']['avg']['sd'][ysel] \
    / numpy.sqrt(zink_data['tilt']['N'][ysel])
# Draw the actual data with error bars.
ax.errorbar(x, y, xerr=xsem, yerr=ysem, fmt='o', \
    color=PLOTCOLS['combined'], ecolor=PLOTCOLS['error'], label='torsion - tilt')
# Set the GVS intensities for the modelling.
s = numpy.arange(GVSAXLIM[0], GVSAXLIM[1], 0.001)
# Compute the functions for the best fitting parameters.
x_lin = vest_func(s, param['torsion']['lin-prop']['a'], \
    param['torsion']['lin-prop']['b'], model='lin-prop', measure='torsion')
x_exp = vest_func(s, param['torsion']['exp']['a'], \
    param['torsion']['exp']['b'], model='exp', measure='torsion')
y_lin = vest_func(s, param['tilt']['lin-prop']['a'], \
    param['tilt']['lin-prop']['b'], model='lin-prop', measure='tilt')
y_exp = vest_func(s, param['tilt']['exp']['a'], \
    param['tilt']['exp']['b'], model='exp', measure='tilt')
# Draw the combinations of the best fits.
ax.plot(x_lin, y_lin, ':', lw=2, color=PLOTCOLS['combined'], label='lin - lin')
ax.plot(x_exp, y_lin, '-.', lw=2, color=PLOTCOLS['combined'], label='exp - lin')
ax.plot(x_exp, y_exp, '--', lw=2, color=PLOTCOLS['combined'], label='exp - exp')
# Finish the figure.
ax.set_xlim([0, 4])
ax.set_xlabel("Ocular torsion (deg)", fontsize=FONTSIZE['axis'])
ax.set_ylim(TAXLIM)
ax.set_ylabel("Visual tilt (deg)", fontsize=FONTSIZE['axis'])
ax.legend(loc='lower right')
# Save the figure.
fig.savefig(os.path.join(OUTDIR, "torsion_x_tilt.%s" % (FIGEXT)))
# Close the figure.
pyplot.close(fig)


# Directional plot, showing the extend to which left and right GVS change
# visual tilt and ocular torsion. It's intended to be a descriptive plot.
print("Plotting directional plot.")
fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(FIGSIZE[1], FIGSIZE[0]), \
    dpi=FIGDPI)
# Loop through the data.
measures = zink_data.keys()
measures.sort()
for m in measures:
    # Get the y-values, which in this case are the stimulation intensities.
    y = zink_data[m]['gvs_intensity']
    # Go through the left and right values.
    for direction in ['left', 'right']:
        # Left stimulation values should be plotted on the left, and thus
        # multiplied by -1.
        if direction is 'left':
            w = -1
        else:
            w = 1
        # Get the x values, which in this case are the tilt or torsion values.
        x = zink_data[m][direction]['m'] * w
        # Get the shading values by adding the standard deviation to the mean.
        x1 = x - zink_data[m][direction]['sd']
        x2 = x + zink_data[m][direction]['sd']
        # Get the light shading values by subtracting or adding the min/max.
        xmin = x - zink_data[m][direction]['min']
        xmax = x + zink_data[m][direction]['max']
        # Plot the average tilt/torsion.
        ax.plot(x, y, '-', \
            lw=LINEWIDTH, color=PLOTCOLS[m], label='%s (%s)' % (m, direction))
        # Plot the shaded areas.
        ax.fill_betweenx(y, x1, x2, color=PLOTCOLS[m], alpha=SDALPHA)
        ax.fill_betweenx(y, xmin, xmax, color=PLOTCOLS[m], alpha=MALPHA)
# Draw the legend.
ax.legend(loc='best', fontsize=FONTSIZE['legend'])
# Set the axis limits.
ax.set_xlim([TAXLIM[1]*-1, TAXLIM[1]])
ax.set_ylim(GVSAXLIM)
# Write the axis labels.
ax.set_xlabel("Ocular torsion or visual tilt (deg)", fontsize=FONTSIZE['axis'])
ax.set_ylabel("Galvanic vestibular stimulation (mA)", fontsize=FONTSIZE['axis'])
# Save the plot.
fig.savefig(os.path.join(OUTDIR, 'directional_plot.%s' % (FIGEXT)))
# Close the plot.
pyplot.close(fig)

