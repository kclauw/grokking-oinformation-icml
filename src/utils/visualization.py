import os 
from dotmap import DotMap
import utils.misc as utils
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from collections import defaultdict
from copy import deepcopy

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}


def lineplot(df, x, y, hue, dict_legend_loss = None, hue_order = None, phase_label_positions = None, hide_legend = False, legend_title = None, show = False, folder = None, filename = None, title = None, xlabel=None, ylabel=None, logscale=False, line_coords = [], show_ylabel = True):
    #sns.set(style="whitegrid")
    style = {'axes.grid' : False,
                            'axes.edgecolor':'black',
                            'family' : 'normal',
                            'weight' : 'bold',
                            'size' : 22,
                            'font_scale': 1.5,
                            "font.size": 22

                            }
    
    sns.set_style("whitegrid", style)
    
    if logscale:
        plt.xscale('log')
        #filename += "_logscale"
   
    
    df["metric_value"] = df["metric_value"].fillna(0)
   
    df = df.reset_index(drop=True)
   
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, errorbar=("se"), hue_order = hue_order)
    
    ax.set_xmargin(0)
    #plt.tight_layout(pad=0)
    # Turns off grid on the left Axis.
    ax.grid(False)
    #plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=2, handletextpad=0.01)
    
    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(weight='bold',
                                   style='normal', size=16)
    
    if dict_legend_loss:
        plt.legend(** dict_legend_loss, prop=font)
    else:
        plt.legend(loc = "center left", prop=font)
    """
    for phase in phase_label_positions:
        phase_label, phase_x = phase
        print((phase_label, phase_x))
        #plt.annotate(p, xy=(p, 10), xycoords='axes fraction', ha='center')
        plt.annotate(phase_label, xy=(phase_x, plt.ylim()[0]), ha='center')
        #plt.text(int(phase_x), 10, phase_label, ha="center", va="center")
    """
    if legend_title is not None:
        ax.get_legend().set_title(legend_title)
    else:
        ax.get_legend().set_title("")
    
    if hide_legend:
        plt.legend([],[], frameon=False)
    

    
   
    ax.set(ylabel=None)

    _store_create_plots(title = title, xlabel = xlabel, ylabel = ylabel, folder = folder, show = show, filename = filename, line_coords = line_coords)
    
def scatterplot(df, x, y, hue = None, show = False, folder = None, filename = None, title = None, xlabel=None, ylabel=None, logscale=False, line_coords = [], show_ylabel = True):
    if hue is not None:
        ax = sns.scatterplot(data=df, x=x, y=y, hue=hue)
    else:
        ax = sns.scatterplot(data=df, x=x, y=y)
        
    ax.get_legend().set_title("")
    
    ax.set(ylabel=None)
    
    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(weight='bold',
                                   style='normal', size=18)
    
    plt.legend(prop=font)
  
    _store_create_plots(title = title, xlabel = xlabel, ylabel = ylabel, folder = folder, show = show, filename = filename)

def sp(x, y, hue_order = None, symbol_mapping = None, color_mapping = None, hue = None, show = False, folder = None, filename = None, title = None, xlabel=None, ylabel=None, huelabel=None, line_coords = None):
    
    
    try:
        c = [color_mapping[category] for category in hue]
    except KeyError as e:
        print(f"Category {e} not found in color_mapping")
  
    #fig, ax = plt.subplots()

    plt.scatter(x, y, c=c, s=20, alpha=0.85, zorder=1, label=c)
  
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
   # Extract unique values from hue
    unique_hue = np.unique(hue)

    # Create legend
    
    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(weight='bold',
                                   style='normal', size=18)
  
    if color_mapping is not None and hue is not None:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[color], markersize=5, label=color) for color in unique_hue]
        plt.legend(handles=handles, title=huelabel, prop=font)
 
    #plt.legend()
       
    
    
    
    #plt.colorbar(label=huelabel)
    _store_create_plots(title = title, xlabel = xlabel, ylabel = ylabel, folder = folder, show = show, filename = filename)

    
def heatmap(data, x = None, y = None, z = None, show = False, folder = None, filename = None, title = None, x_label=None, y_label=None):

    if isinstance(data, pd.DataFrame):
        data_matrix = data.pivot(index=x, columns=y, values=z)
        data_matrix = data_matrix.apply(pd.to_numeric, errors='coerce')
    else:
        data_matrix = data
    
    s = sns.heatmap(data_matrix, annot=True, fmt=".1f", linewidths=.5,
                    annot_kws={'size': 10}, mask=data_matrix.isnull())
    
    if x_label:
        s.set_ylabel(x_label, fontsize=10)
    if y_label:
        s.set_xlabel(y_label, fontsize=10)
  
    """
    fmt = '.2f' if np.issubdtype(data_matrix, np.floating) else 'd'
    ax = sns.heatmap(data_matrix, annot=True, cmap='viridis', fmt=fmt, linewidths=.5,
                    annot_kws={'size': 10}) 
   
    plt.imshow(data, cmap="hot", interpolation="nearest", aspect="auto")
    plt.colorbar()
    #ax.get_legend().set_title("")
    
    #ax.set(ylabel=None)
    """
    _store_create_plots(title = title, folder = folder, show = show, filename = filename)
    

def _store_create_plots(title = None, xlabel = None, ylabel = None, folder = None, show = None, filename = None, line_coords = None):
    
    if line_coords is not None:
        for xc in line_coords:
            plt.axvline(x=xc, color='black', linestyle ='--')
            
      
    
    

    if title is not None:
        plt.title(title)
        
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=20, weight='bold')
        
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=20, weight='bold')
    plt.tight_layout(pad=0)
    
    if folder is not None:    
        utils.create_folder(folder)
        plt.savefig(os.path.join(folder, filename + '.pdf'))
    
    if show:
        plt.show()
        
    plt.close()
