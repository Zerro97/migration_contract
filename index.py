import numpy as np
import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter

from scipy.interpolate import make_interp_spline, BSpline
from scipy.stats import gaussian_kde
from scipy.optimize import minimize, Bounds
from sympy import symbols, Eq, solve

## Solving for two-period model without migration

# Initializing variables
#s_a = 11
s_w = 11
s_h = 11
alpha = np.linspace(0, 1, s_a)
beta = 0.98
eta = 0.7
theta1 = 1
theta2 = 2
w = np.linspace(0, 10, s_w)
sigma = 2
tau1 = 0
tau2 = 0
h1 = np.linspace(0, 1, s_h)
nu_prime = np.zeros((s_a,s_w,s_h))

def u_nomigration(nu, alpha, w, h1):
    c1 = w + (1-nu) * (1-tau1) * theta1 * (h1**alpha)
    c2 = (1-tau2) * theta2 * (h1**alpha) * ((1 + eta*nu)**alpha)
    return -(c1**(1 - sigma)) / (1 - sigma) - beta * (c2**(1 - sigma)) / (1 - sigma)

print('hi')

for i in range(0, s_a):
    for j in range(0, s_w):
        for k in range(0, s_h):
            #print("alpha = ", alpha[i])
            #print("initial wealth = ", w[j])
            #print("initial human capital = ", h1[k])
            #eq1 = Eq(alpha[i] * beta * eta * pow((1 - tau2) * theta2 * pow(h1[k],alpha[i]),1 - sigma) * pow(1 +                     eta * nu, alpha[i] * (1 - sigma) - 1) - pow(w[j] + (1 - nu) * (1 - tau1) * theta1 *                               pow(h1[k],alpha[i]),-sigma) * (1 - tau1) * theta1 * pow(h1[k],alpha[i]))   
            
            #print('hi')
            nu0 = 0.5
            bounds = Bounds(0, 1)
            #sol = minimize(u_nomigration, nu0, bounds = bounds, method = 'trust-constr', args = (alpha[i], w[j], h1[k]))
            #nu_prime[i,j,k] = sol.x
            
# Write the array to disk
'''
with open('test.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(nu_prime.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in nu_prime:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice, fmt='%-7.2f')

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')
'''
            

'''
## Artuc et. all. net high-skilled emigration rates

x = np.arange(6)
net_emi = [-5.5, -6.6, 8.0, 16.0, 16.6, 34.8]

fig, ax = plt.subplots()
plt.bar(x, net_emi, alpha = 0.5, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'])
plt.axhline(y=0, color = 'black', linestyle='-', linewidth = 1, alpha = 0.5)
plt.ylabel("Net high-skilled emigration rates (%)")
plt.xticks(x, ('OECD', 'HIGH', 'DEV', 'LOW', 'LDC', 'SIDS'))
out_png = '/workspace/data_contractual/net_emigration2000.png'
plt.savefig(out_png, dpi=150,bbox_inches = 'tight',
    pad_inches = 0.1)
plt.show()

gross_emi = [4.8, 4.8, 12.0, 20.3, 19.9, 40.9]

fig, ax = plt.subplots()
plt.bar(x, gross_emi, alpha = 0.5, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'])
plt.ylabel("Gross high-skilled emigration rates (%)")
plt.xticks(x, ('OECD', 'HIGH', 'DEV', 'LOW', 'LDC', 'SIDS'))
out_png = '/workspace/data_contractual/gross_emigration2000.png'
plt.savefig(out_png, dpi=150,bbox_inches = 'tight',
    pad_inches = 0.1)
plt.show()

## UNESCO UIS data on international student mobility
data = pd.read_csv('/workspace/data_contractual/UNESCO_student_mobility/UNESCO_student_mobility.csv')
world = data[(data['EDULIT_IND'] == 'OE_5T8_40510') & (data['LOCATION'].str.len() == 5)]
indicator = world.LOCATION.unique()
income_country = ['Low income countries', 'Lower middle income countries', 'Middle income countries', 'Upper middle income countries', 'High income countries']
income_location = ['40044','40042','40030','40043','40041']
region_country = ['Arab States','Central and Eastern Europe','Central Asia','East Asia and the Pacific','Latin America and the Caribbean','North America and Western Europe','South and West Asia','Sub-Saharan Africa','Small Island Developing States']
region_location = ['40525','40530','40505','40515','40520','40500','40535','40540','40550']
income = pd.DataFrame({'location':income_location, 'country':income_country})
region = pd.DataFrame({'location':region_location, 'country':region_country})
time = world.Time.unique()

fig, ax = plt.subplots()
for index, row in income.iterrows():
    temp = world.loc[world['LOCATION'] == row['location']]
    plt.plot(temp.Time,temp.Value, label=row['country'], alpha = 0.7)
plt.ylabel("Outbound tertiary students studying abroad (thousands)")
plt.xlabel("Year")
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000)))
plt.legend(loc='upper left')
out_png = '/workspace/data_contractual/UNESCO_income.png'
plt.savefig(out_png, dpi=150,bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots()
for index, row in region.iterrows():
    temp = world.loc[world['LOCATION'] == row['location']]
    plt.plot(temp.Time,temp.Value, label=row['country'], alpha = 0.7)
plt.ylabel("Outbound tertiary students studying abroad (thousands)")
plt.xlabel("Year")
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000)))
ax.legend(loc='upper center', bbox_to_anchor=(0, -0.6,1, 0.5),
          ncol=2, mode="expand",prop={'size': 8})
out_png = '/workspace/data_contractual/UNESCO_region.png'
plt.savefig(out_png, dpi=150,bbox_inches = 'tight',
    pad_inches = 0)
'''
'''
## Polarization of the Public
# ANES Data 2016
print("ANES 2016")
df_ANES = pd.read_stata('/workspace/Political_Polarizati/ANES/2016/anes_timeseries_2016.dta')
var = ['V161178','V161181','V161184','V161187','V161189','V161192','V161195x','V161196x','V161198','V161201','V161204x','V161205','V161206','V161206','V161209','V161211','V161212','V161213x','V161214x','V161225x','V161227x','V161228x','V161229x','V161230','V161231','V161232','V161158x','V160101']
var1 = ['V161178','V161181','V161184','V161187','V161189','V161192','V161195x','V161196x','V161198','V161201','V161204x','V161205','V161206','V161206','V161209','V161211','V161212','V161213x','V161214x','V161225x','V161227x','V161228x','V161229x','V161230','V161231','V161232','V161158x']
var2 = ['V161178','V161181','V161184','V161189','V161198','V161201']
df = df_ANES[var]
df_ANES[var2] = df_ANES[var2].astype(str)  
for i in var1:
    df[i] = df_ANES[i].str[0]
df = df.replace('-', np.nan)
df = df[var].astype(str).astype(float)
df = df.loc[:,~df.columns.duplicated()]

# Recoding variable such that higher number means more conservative
dict7 = {7:1,6:2,5:3,4:4,3:5,2:6,1:7}
dict7correct = {1:1,2:2,3:3,4:4,5:5,6:6,7:7}
dict3inv = {1:1,2:3,3:2}
dict4 = {1:4,2:3,3:2,4:1}
dict4correct = {1:1,2:2,3:3,4:4}
dict6 = {1:6,2:5,3:4,4:3,5:2,6:1}
dict2 = {1:2,2:1}
dict3correct = {1:1,2:2,3:3}
for i in ['V161178','V161213x','V161196x']:
    df[i] = df[i].map(dict7)
for i in ['V161181','V161184','V161189','V161198','V161201','V161204x','V161214x','V161225x']:
    df[i] = df[i].map(dict7correct)
for i in ['V161187','V161205','V161206','V161209','V161211','V161212']:
    df[i] = df[i].map(dict3inv)
for i in ['V161192','V161232']:
    df[i] = df[i].map(dict4)
df['V161229x'] = df.V161229x.map(dict4correct)
for i in ['V161195x','V161227x','V161228x']:
    df[i] = df[i].map(dict6)
df['V161230'] = df.V161230.map(dict2)
df['V161231'] = df.V161231.map(dict3correct)
df['V161158x'] = df['V161158x'].replace([0,1,2],1) # Democrats
df['V161158x'] = df['V161158x'].replace([4,5,6],2) # Republicans
df_id = df[['V161158x']] # 1 = Democrat, 2 = Republican
df_weight = df[['V160101']]
df = df.drop(['V161158x','V160101'], axis=1)
print(df.isna().sum())
df = df.dropna()
df = (df - df.mean())/df.std()

# Factor Analysis
chi_square_value,p_value=calculate_bartlett_sphericity(df)
print(chi_square_value, p_value)

kmo_all,kmo_model=calculate_kmo(df)
print(kmo_model)

fa = FactorAnalyzer()
fa.analyze(df, 10, rotation=None)
ev, v = fa.get_eigenvalues()
print(ev)

fa.analyze(df, 1, rotation="promax")
print(fa.loadings)
print(fa.get_factor_variance())
factor_load = fa.loadings

# Index constructed by simply adding the normalized variables
print("Index1")
index1 = df.sum(axis=1)
index1 = pd.Series.to_frame(index1)
index1 = (index1 - index1.mean())/index1.std()
index1['id'] = df_id
index1['weights'] = df_weight
index1 = index1.rename(columns={index1.columns[0]: "index" })
index_D = index1.loc[index1['id'] == 1][['index','weights']]   
index_R = index1.loc[index1['id'] == 2][['index','weights']]  
med_index_D = index_D['index'].median()
med_index_R = index_R['index'].median()
left_R = index_R.apply(lambda x: True if x['index'] < med_index_D else False , axis=1)
right_D = index_D.apply(lambda x: True if x['index'] > med_index_R else False , axis=1)
numLeft_R = len(left_R[left_R == True].index)/len(index_R.index)
numRight_D = len(right_D[right_D == True].index)/len(index_D.index)
print("Percentage of Republican who are more liberal than median Democrats in 2016: ", numLeft_R)
print("Percentage of Democrats who are more conservative than median Republicans in 2016: ", numRight_D)

density_D = gaussian_kde(index_D['index'],weights=index_D['weights'])
x_D = np.linspace(np.amin(index_D['index']),np.amax(index_D['index']),500)
density_D.covariance_factor = lambda : .25
density_D._compute_covariance()

density_R = gaussian_kde(index_R['index'],weights=index_R['weights'])
x_R = np.linspace(np.amin(index_R['index']),np.amax(index_R['index']),500)
density_R.covariance_factor = lambda : .25
density_R._compute_covariance()

fig, ax = plt.subplots()
ax.fill_between(x_D, density_D(x_D), 0, facecolor='b', alpha=0.3)
ax.fill_between(x_R, density_R(x_R), 0, facecolor='orangered', alpha=0.3)
p3 = plt.axvline(x=med_index_D,color='b',ls='--')
p4 = plt.axvline(x=med_index_R,color='orangered',ls='--')
ax.margins(0)
plt.ylabel("Density")
plt.xlabel("Individual Ideology")
plt.title("Fig 4b: Distribution of Ideology in the Public in 2016")
p1 = ax.fill(np.NaN, np.NaN, 'b', alpha=0.3)
p2 = ax.fill(np.NaN, np.NaN, 'orangered', alpha=0.3)
ax.legend([(p1[0]),(p2[0]), p3, p4], ['Democrats','Republicans','Med. Democrats','Med. Republicans'],loc=1)
out_png = '/workspace/Political_Polarizati/ANES/2016/public_index1_16.png'
plt.savefig(out_png, dpi=150)
plt.close()

# Index created by factor analysis
print("Index2")
index2 = df.dot(-fa.loadings)
index2['id'] = df_id
index2['weights'] = df_weight
index2 = index2.rename(columns={index2.columns[0]: "index" })
index_D = index2.loc[index2['id'] == 1][['index','weights']]   
index_R = index2.loc[index2['id'] == 2][['index','weights']]  
med_index_D = index_D['index'].median()
med_index_R = index_R['index'].median()
left_R = index_R.apply(lambda x: True if x['index'] < med_index_D else False , axis=1)
right_D = index_D.apply(lambda x: True if x['index'] > med_index_R else False , axis=1)
numLeft_R = len(left_R[left_R == True].index)/len(index_R.index)
numRight_D = len(right_D[right_D == True].index)/len(index_D.index)
print("Percentage of Republican who are more liberal than median Democrats in 2016: ", numLeft_R)
print("Percentage of Democrats who are more conservative than median Republicans in 2016: ", numRight_D)

density_D = gaussian_kde(index_D['index'],weights=index_D['weights'])
x_D = np.linspace(np.amin(index_D['index']),np.amax(index_D['index']),500)
density_D.covariance_factor = lambda : .25
density_D._compute_covariance()

density_R = gaussian_kde(index_R['index'],weights=index_R['weights'])
x_R = np.linspace(np.amin(index_R['index']),np.amax(index_R['index']),500)
density_R.covariance_factor = lambda : .25
density_R._compute_covariance()

fig, ax = plt.subplots()
ax.fill_between(x_D, density_D(x_D), 0, facecolor='b', alpha=0.3)
ax.fill_between(x_R, density_R(x_R), 0, facecolor='orangered', alpha=0.3)
p3 = plt.axvline(x=med_index_D,color='b',ls='--')
p4 = plt.axvline(x=med_index_R,color='orangered',ls='--')
ax.margins(0)
plt.ylabel("Density")
plt.xlabel("Individual Ideology")
plt.title("Fig 3b: Distribution of Ideology in the Public in 2016")
p1 = ax.fill(np.NaN, np.NaN, 'b', alpha=0.3)
p2 = ax.fill(np.NaN, np.NaN, 'orangered', alpha=0.3)
ax.legend([(p1[0]),(p2[0]), p3, p4], ['Democrats','Republicans','Med. Democrats','Med. Republicans'],loc=1)
out_png = '/workspace/Political_Polarizati/ANES/2016/public_index2_16.png'
plt.savefig(out_png, dpi=150)
plt.close()
'''