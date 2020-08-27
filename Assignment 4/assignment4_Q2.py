# load necessary libraries
import pandas
import numpy
import scipy
from sklearn import naive_bayes

print()
print("=" * 50)
print("=" * 50)
print("ML-Assignment 4-Question 2")
print("=" * 50)
print("=" * 50)
# Please answer the following questions based on your model.


# define a function to visualize the percent of a particular target category by a nominal predictor
def row_with_column(rowVar, columnVar, show='ROW'):
    countTable = pandas.crosstab(index=rowVar, columns=columnVar, margins=False, dropna=True)
    print("Frequency Table: \n", countTable)
    print()

    if show == 'ROW' or show == 'BOTH':
        rowFraction = countTable.div(countTable.sum(1), axis='index')
        print("Row Fraction Table: \n", rowFraction)
        print()

    if show == 'COLUMN' or show == 'BOTH':
        columnFraction = countTable.div(countTable.sum(0), axis='columns')
        print("Column Fraction Table: \n", columnFraction)
        print()

    return


# define a function that performs the Chi-square test
def chi_square_test(x_cat, y_cat, debug='N'):
    obs_count = pandas.crosstab(index=x_cat, columns=y_cat, margins=False, dropna=True)
    c_total = obs_count.sum(axis=1)
    r_total = obs_count.sum(axis=0)
    n_total = numpy.sum(r_total)
    exp_count = numpy.outer(c_total, (r_total / n_total))

    if debug == 'Y':
        print('Observed Count:\n', obs_count)
        print('Column Total:\n', c_total)
        print('Row Total:\n', r_total)
        print('Overall Total:\n', n_total)
        print('Expected Count:\n', exp_count)
        print('\n')

    chi_sq_stat = ((obs_count - exp_count) ** 2 / exp_count).to_numpy().sum()
    chi_sq_df = (obs_count.shape[0] - 1.0) * (obs_count.shape[1] - 1.0)
    chi_sq_sig = scipy.stats.chi2.sf(chi_sq_stat, chi_sq_df)

    cramer_v = chi_sq_stat / n_total
    if c_total.size > r_total.size:
        cramer_v = cramer_v / (r_total.size - 1.0)
    else:
        cramer_v = cramer_v / (c_total.size - 1.0)
    cramer_v = numpy.sqrt(cramer_v)

    return chi_sq_stat, chi_sq_df, chi_sq_sig, cramer_v


# load data file
purchase_likelihood = pandas.read_csv('Purchase_Likelihood.csv', delimiter=',',
                                      usecols=['group_size', 'homeowner', 'married_couple', 'A'])
purchase_likelihood = purchase_likelihood.dropna()

# create category and interval variable list
predictor = ['group_size', 'homeowner', 'married_couple']

print()
print("=" * 50)
print("ML-Assignment 4-Question 2-Section a)")
print("=" * 50)
# a)	(5 points) Show in a table the frequency counts and the Class Probabilities of the target variable.
freq = purchase_likelihood.groupby('A').size()
table = pandas.DataFrame(columns=['count', 'class probability'])
table['count'] = freq
table['class probability'] = table['count'] / purchase_likelihood.shape[0]
print(table)

print()
print("=" * 50)
print("ML-Assignment 4-Question 2-Section b)")
print("=" * 50)
# b)	(5 points) Show the crosstabulation table of the target variable by the feature group_size.  The table
# contains the frequency counts.
row_with_column(purchase_likelihood['A'], purchase_likelihood['group_size'], 'ROW')

print()
print("=" * 50)
print("ML-Assignment 4-Question 2-Section c)")
print("=" * 50)
# c)	(5 points) Show the crosstabulation table of the target variable by the feature homeowner.  The table contains
# the frequency counts.
row_with_column(purchase_likelihood['A'], purchase_likelihood['homeowner'], 'ROW')

print()
print("=" * 50)
print("ML-Assignment 4-Question 2-Section d)")
print("=" * 50)
# d)	(5 points) Show the crosstabulation table of the target variable by the feature married_couple.  The table
# contains the frequency counts.
row_with_column(purchase_likelihood['A'], purchase_likelihood['married_couple'], 'ROW')

print()
print("=" * 50)
print("ML-Assignment 4-Question 2-Section e)")
print("=" * 50)
# e)	(10 points) Calculate the Cramer’s V statistics for the above three crosstabulations tables.  Based on these
# Cramer’s V statistics, which feature has the largest association with the target A?
test_result = pandas.DataFrame(index=predictor,
                               columns=['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])
for pred in predictor:
    chi_sq_stat, chi_sq_df, chi_sq_sig, cramer_v = chi_square_test(purchase_likelihood[pred], purchase_likelihood['A'],
                                                                   debug='N')
    test_result.loc[pred] = ['Chi-square', chi_sq_stat, chi_sq_df, chi_sq_sig, 'Cramer''V', cramer_v]
rank_assoc = test_result.sort_values('Measure', axis=0, ascending=False)
print(rank_assoc)

# train naive baues model
xTrain = purchase_likelihood[predictor].astype('category')
yTrain = purchase_likelihood['A'].astype('category')

_objNB = naive_bayes.MultinomialNB(alpha=1.0e-10)
thisFit = _objNB.fit(xTrain, yTrain)

print()
print("=" * 50)
print("ML-Assignment 4-Question 2-Section g)")
print("=" * 50)
# g)	(10 points) For each of the sixteen possible value combinations of the three features, calculate the predicted
# probabilities for A = 0, 1, 2 based on the Naïve Bayes model.  List your answers in a table with proper labelling.
# Create the all possible combinations of the features' values
gs_d = [1, 2, 3, 4]
ho_d = [0, 1]
mc_d = [0, 1]
A_d = [0, 1, 2]
final_data = []
for gsd in gs_d:
    for hod in ho_d:
        for mcd in mc_d:
            data = [gsd, hod, mcd]
            final_data = final_data + [data]
x_test = pandas.DataFrame(final_data, columns=['group_size', 'homeowner', 'married_couple'])
x_test = x_test[predictor].astype('category')
# predict probabities for all combinations
y_test_pred_prob = pandas.DataFrame(_objNB.predict_proba(x_test), columns=['p_a_0', 'p_a_1', 'p_a_2'])
y_test_score = pandas.concat([x_test, y_test_pred_prob], axis=1)
print(y_test_score)

print()
print("=" * 50)
print("ML-Assignment 4-Question 2-Section g)")
print("=" * 50)
# h)	(5 points) Based on your model, what values of group_size, homeowner, and married_couple will maximize the
# odds value Prob(A=1) / Prob(A = 0)?  What is that maximum odd value?
y_test_score['odd value(p_a_1/p_a_0)'] = y_test_score['p_a_1'] / y_test_score['p_a_0']
print(y_test_score[['group_size', 'homeowner', 'married_couple', 'odd value(p_a_1/p_a_0)']])
print(y_test_score.loc[y_test_score['odd value(p_a_1/p_a_0)'].idxmax()])
