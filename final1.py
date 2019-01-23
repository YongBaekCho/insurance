import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt

def read_file():
    '''
    This function does not take any parameter.
    It returns DataFrame created by read_csv.
    '''
    df = pd.read_csv("insurance.csv")

    for row in df.index:
        for col in df.columns:
            if df.loc[row,'bmi'] < 18.5:
                df.loc[row,'level'] = 'Under weight'
            elif df.loc[row,'bmi'] < 25:
                df.loc[row,'level'] = 'Normal range'
            elif df.loc[row,'bmi'] < 30:
                df.loc[row,'level'] = 'Over weight'
            else:
                df.loc[row,'level'] = 'Obese'
    return df

def age_bmi_data(df):

    column = ['teen', 'twenty', 'thirty', 'fourty', 'fifty', 'sixty']
    index = ['male', 'female']
    count = pd.DataFrame(np.zeros((2,6)), index = index, columns = column)
    data = pd.DataFrame(np.zeros((2,6)), index = index, columns = column)

    for row in df.index:
        if df.loc[row, 'age'] < 20:
            if df.loc[row, 'sex'] == 'male':
                data.loc['male', 'teen'] += df.loc[row, 'bmi']
                count.loc['male', 'teen'] += 1
            else:
                data.loc['female', 'teen'] += df.loc[row, 'bmi']
                count.loc['female', 'teen'] += 1
        elif df.loc[row, 'age'] < 30:
            if df.loc[row, 'sex'] == 'male':
                data.loc['male', 'twenty'] += df.loc[row, 'bmi']
                count.loc['male', 'twenty'] += 1
            else:
                data.loc['female', 'twenty'] += df.loc[row, 'bmi']
                count.loc['female', 'twenty'] += 1
        elif df.loc[row, 'age'] < 40:
            if df.loc[row, 'sex'] == 'male':
                data.loc['male', 'thirty'] += df.loc[row, 'bmi']
                count.loc['male', 'thirty'] += 1
            else:
                data.loc['female', 'thirty'] += df.loc[row, 'bmi']
                count.loc['female', 'thirty'] += 1
        elif df.loc[row, 'age'] < 50:
            if df.loc[row, 'sex'] == 'male':
                data.loc['male', 'fourty'] += df.loc[row, 'bmi']
                count.loc['male', 'fourty'] += 1
            else:
                data.loc['female', 'fourty'] += df.loc[row, 'bmi']
                count.loc['female', 'fourty'] += 1
        elif df.loc[row, 'age'] < 60:
            if df.loc[row, 'sex'] == 'male':
                data.loc['male', 'fifty'] += df.loc[row, 'bmi']
                count.loc['male', 'fifty'] += 1
            else:
                data.loc['female', 'fifty'] += df.loc[row, 'bmi']
                count.loc['female', 'fifty'] += 1
        else:
            if df.loc[row, 'sex'] == 'male':
                data.loc['male', 'sixty'] += df.loc[row, 'bmi']
                count.loc['male', 'sixty'] += 1
            else:
                data.loc['female', 'sixty'] += df.loc[row, 'bmi']
                count.loc['female', 'sixty'] += 1
    
    for row in data.index:
        for col in data.columns:
            data.loc[row, col] = data.loc[row, col] / count.loc[row, col]

    return data

def age_bmi_plt(data):

    n_groups = 6

    means_male = data.loc['male']
    means_female = data.loc['female']

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    male = plt.bar(index + bar_width, means_male, bar_width, color = 'royalblue', label = 'Male')
    female = plt.bar(index, means_female, bar_width, color = 'lightcoral', label = 'Female')    
    
    plt.xlabel('Age range', fontsize = 24)
    plt.ylabel('BMI', fontsize = 24)
    plt.title('BMI by age range and gender', fontsize = 24)
    plt.xticks(index + bar_width / 2, data.columns)
    plt.legend()

def charge_mean(df):

    index = [10, 20, 30, 40, 50, 60]
    count = pd.Series(np.zeros(6), index = index)
    data = pd.Series(np.zeros(6), index = index)

    for row in df.index:
        if df.loc[row, 'age'] < 20:
            data[10] += df.loc[row, 'charges']
            count[10] += 1

        elif df.loc[row, 'age'] < 30:
            data[20] += df.loc[row, 'charges']
            count[20] += 1

        elif df.loc[row, 'age'] < 40:
            data[30] += df.loc[row, 'charges']
            count[30] += 1

        elif df.loc[row, 'age'] < 50:
            data[40] += df.loc[row, 'charges']
            count[40] += 1

        elif df.loc[row, 'age'] < 60:
            data[50] += df.loc[row, 'charges']
            count[50] += 1

        else:
            data[60] += df.loc[row, 'charges']
            count[60] += 1
    
    for row in data.index:
        data.loc[row] = data[row] / count[row]

    return data

def charge_age_plt(ser, df):

    age_array = sm.add_constant(ser.index)
    model = sm.OLS(ser.values, age_array)
    results = model.fit()
    slope = results.params[1]
    intercept = results.params[0]
    r2 = results.rsquared
    p_val = results.pvalues[0]

    df.plot(kind='scatter', x = 'age', y = 'charges', c = 'bmi', colormap = 'Purples')
    xs = np.arange(df['age'].min(), df['age'].max())
    ys = slope * xs + intercept
    plt.plot(xs, ys)

    ax = plt.gca()
    ax.set_ylabel('Charges', fontsize = 24)
    ax.set_xlabel('Age', fontsize = 24)

    plt.title('Charges vs. Age', fontsize = 24)

def charge_bmi(df):

    search = ['Under weight', 'Normal range', 'Over weight', 'Obese']
    a = pd.Series(np.zeros(4), index = search)
    b = pd.Series(np.zeros(4), index = search)
    
    
    for row in df.index:
        if df.loc[row, 'bmi'] < 18.5:
            a['Under weight'] += df.loc[row, 'charges']
            b['Under weight'] += 1
        elif df.loc[row, 'bmi'] < 25:
            a['Normal range'] += df.loc[row, 'charges']
            b['Normal range'] += 1
        elif df.loc[row, 'bmi'] < 30:
            a['Over weight'] += df.loc[row, 'charges']
            b['Over weight'] += 1
        else:
            a['Obese'] += df.loc[row, 'charges']
            b['Obese'] += 1
    for row in a.index:
        a.loc(axis=0)[row] = a[row] / b[row]
    return a
    
def bmi_charges_plt(f1, df):
    plt.figure()
    plt.plot(f1, linestyle = 'dashed', color = 'red', label = 'mean by interval')
    
    ax = plt.gca()
    ax.set_ylabel('Charges mean', fontsize = 24)
    ax.set_xlabel('BMI Interval', fontsize = 24)
    
    ax.yaxis.label.set_fontsize(24)
    
    ax.set_xticklabels(['Under weight', 'Normal range', 'Over weight', 'Obese'])
    ax.set_title('Charges mean by BMI', fontsize = 24)
    plt.legend()


def obese_by_sex(df):

    obj = ['Under weight', 'Normal range', 'Over weight', 'Obese']
    y_pos = np.arange(len(obj))
    female = np.arange(len(obj))
    female = female.astype(float)
    male = np.arange(len(obj))
    male = male.astype(float)
    plt.figure()

    for row in df.index:
        if df.loc[row, 'sex'] == 'male':
            if df.loc[row,'level'] == obj[0]:
                male[0] += 1
            elif df.loc[row,'level'] == obj[1]:
                male[1] += 1
            elif df.loc[row,'level'] == obj[2]:
                male[2] += 1
            else:
                male[3] += 1
    for row in df.index:
        if df.loc[row, 'sex'] == 'female':
            if df.loc[row,'level'] == obj[0]:
                female[0] += 1
            elif df.loc[row,'level'] == obj[1]:
                female[1] += 1
            elif df.loc[row,'level'] == obj[2]:
                female[2] += 1
            else:
                female[3] += 1

    male_bar = plt.bar(y_pos, +male, facecolor='#9999ff', edgecolor='white', width = 0.5)
    female_bar = plt.bar(y_pos, -female, facecolor='#ff9999', edgecolor='white', width = 0.5)
    plt.xticks(y_pos, obj)
    plt.ylim(-450,450)

    ax = plt.gca()
    ax.set_xlabel('BMI Categories', fontsize = 24)
    ax.set_ylabel('BMI Interval', fontsize = 24)
    ax.set_title("BMI by Sex", fontsize = 24)

    for x,y in zip(np.arange(len(obj)), male):
        plt.text(x, y, '%d' % y, ha = 'center', va = 'bottom', color = '#9999ff')
    for x,y in zip(np.arange(len(obj)), female):
        plt.text(x, -y, '%d' % y, ha = 'center', va = 'top', color = '#ff9999')

    return male, female

def pie_chart(male, female):

    m_tot, f_tot = 0.0, 0.0
    m_val = ['Under weight', 'Normal range', 'Over weight', 'Obese']
    f_val = ['Under weight', 'Normal range', 'Over weight', 'Obese']
    
    for m in male:
        m_tot += m
    for f in female:
        f_tot += f

    for i in range(len(male)):
        male[i] = (male[i] / m_tot) * 100
    for j in range(len(female)):
        female[i] = (female[i] / f_tot) * 100

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    pie_1 = axes[0].pie(male,autopct='%1.1f%%')
    
    axes[0].set_title("BMI percentage of Male", fontsize = 24)
    axes[0].legend(m_val, loc = 'bottm left')
    pie_2 = axes[1].pie(female, autopct='%1.1f%%', startangle=90, pctdistance = 1.19, labeldistance = 1.3)
    axes[1].set_title("BMI percentage by Female", fontsize = 24)

def region_bmi(df):

    plt.figure()
    
    data = pd.DataFrame(0, index = ['northeast', 'northwest', 'southeast', 'southwest'], columns = ['region_total', 'high_bmi'])
    
    for row in df.index:
        if df.loc[row,'region'] == 'northeast':
            data.loc['northeast', 'region_total'] += 1
        elif df.loc[row,'region'] == 'northwest':
            data.loc['northwest', 'region_total'] += 1
        elif df.loc[row,'region'] == 'southeast':
            data.loc['southeast', 'region_total'] += 1
        else:
            data.loc['southwest', 'region_total'] += 1

    for row in df.index:
        if df.loc[row,'level'] == 'Obese':
            if df.loc[row,'region'] == 'northeast':
                data.loc['northeast', 'high_bmi'] += 1
            elif df.loc[row,'region'] == 'northwest':
                data.loc['northwest', 'high_bmi'] += 1
            elif df.loc[row,'region'] == 'southeast':
                data.loc['southeast', 'high_bmi'] += 1
            else:
                data.loc['southwest', 'high_bmi'] += 1
     
    horizontal = plt.barh(np.arange(len(data.index)), (data['high_bmi'] / data['region_total']) * 100)
    plt.yticks(np.arange(len(data.index)), data.index)
    plt.xlim(0,100)

    ax = plt.gca()
    ax.set_xlabel('Number of Obese', fontsize = 24)
    ax.set_ylabel('Region', fontsize = 24)
    ax.set_title("Number of Obese by region", fontsize = 24)

def main():
    df = read_file()
    age_bmi = age_bmi_data(df)
    age_bmi_figure = age_bmi_plt(age_bmi)
    charge_mean_by_age = charge_mean(df)
    charge_age_figure = charge_age_plt(charge_mean_by_age, df)
    obs_male, obs_female = obese_by_sex(df)
    pie = pie_chart(obs_male, obs_female)
    region = region_bmi(df)
    bmi_charges = charge_bmi(df)
    bmi_charges_figure = bmi_charges_plt(bmi_charges, df)

    plt.show()
    
main()