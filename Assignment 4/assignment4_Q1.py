# load necessary libraries
import pandas
import statsmodels.api as stats
import sympy
import scipy
import numpy

print()
print("=" * 50)
print("=" * 50)
print("ML-Assignment 4-Question 1")
print("=" * 50)
print("=" * 50)
# Please answer the following questions based on your model.


# create interaction parameters for model
def create_interaction(in_df1, in_df2):
    name1 = in_df1.columns
    name2 = in_df2.columns
    out_df = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            out_df[outName] = in_df1[col1] * in_df2[col2]
    return out_df


# function that crate mnlogit model and return llk,df,full params
def build_mnlogit(full_x, y, debug='N'):
    no_full_param = full_x.shape[1]

    y_category = y.cat.categories
    no_y_cat = len(y_category)

    reduced_form, inds = sympy.Matrix(full_x.values).rref()

    if debug == 'Y':
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    x = full_x.iloc[:, list(inds)]

    this_df = len(inds) * (no_y_cat - 1)

    logit = stats.MNLogit(y, x)
    this_fit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
    this_parameter = this_fit.params
    this_llk = logit.loglike(this_parameter.values)

    if debug == 'Y':
        print(this_fit.summary())
        print("Model Parameter Estimates:\n", this_parameter)
        print("Model Log-Likelihood Value =", this_llk)
        print("Number of Free Parameters =", this_df)

    work_params = pandas.DataFrame(numpy.zeros(shape=(no_full_param, (no_y_cat - 1))))
    work_params = work_params.set_index(keys=full_x.columns)
    full_params = pandas.merge(work_params, this_parameter, how="left", left_index=True, right_index=True)
    full_params = full_params.drop(columns='0_x').fillna(0.0)

    return this_llk, this_df, full_params


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


# define a function that performs the deviance test
def deviance_test(x_int, y_cat, debug='N'):
    y = y_cat.astype('category')
    x = numpy.where(y_cat.notnull(), 1, 0)
    obj_logit = stats.MNLogit(y, x)
    this_fit = obj_logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
    this_parameter = this_fit.params
    llk0 = obj_logit.loglike(this_parameter.values)

    if debug == 'Y':
        print(this_fit.summary())
        print("Model Log-Likelihood Value =", llk0)
        print('\n')

    x = stats.add_constant(x_int, prepend=True)
    obj_logit = stats.MNLogit(y, x)
    this_fit = obj_logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
    this_parameter = this_fit.params
    llk1 = obj_logit.loglike(this_parameter.values)

    if debug == 'Y':
        print(this_fit.summary())
        print("Model Log-Likelihood Value =", llk1)

    deviance_stat = 2.0 * (llk1 - llk0)
    deviance_df = (len(y.cat.categories) - 1.0)
    deviance_sig = scipy.stats.chi2.sf(deviance_stat, deviance_df)

    mc_fadden_r_sq = 1.0 - (llk1 / llk0)

    return deviance_stat, deviance_df, deviance_sig, mc_fadden_r_sq


# load data file
purchase_likelihood = pandas.read_csv('Purchase_Likelihood.csv', delimiter=',',
                                      usecols=['group_size', 'homeowner', 'married_couple', 'A'])
purchase_likelihood = purchase_likelihood.dropna()
no_objs = purchase_likelihood.shape[0]
y = purchase_likelihood['A'].astype('category')
# create dummy variables
x_gs = pandas.get_dummies(purchase_likelihood[['group_size']].astype('category'))
x_ho = pandas.get_dummies(purchase_likelihood[['homeowner']].astype('category'))
x_mc = pandas.get_dummies(purchase_likelihood[['married_couple']].astype('category'))

# intercept only model
design_x = pandas.DataFrame(y.where(y.isnull(), 1))
llk0, df0, full_params0 = build_mnlogit(design_x, y, debug='Y')

# intercept + group_size
design_x = stats.add_constant(x_gs, prepend=True)
llk1_gs, df1_gs, full_params1_gs = build_mnlogit(design_x, y, debug='Y')
test_dev_gs = 2 * (llk1_gs - llk0)
test_df_gs = df1_gs - df0
test_p_value_gs = scipy.stats.chi2.sf(test_dev_gs, test_df_gs)

# intercept + group_size + homeowner
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = stats.add_constant(design_x, prepend=True)
llk2_gs_ho, df2_gs_ho, full_params2_gs_ho = build_mnlogit(design_x, y, debug='Y')
test_dev_gs_ho = 2 * (llk2_gs_ho - llk1_gs)
test_df_gs_ho = df2_gs_ho - df1_gs
test_p_value_gs_ho = scipy.stats.chi2.sf(test_dev_gs_ho, test_df_gs_ho)

# intercept + group_size + homeowner + married_couple
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
design_x = stats.add_constant(design_x, prepend=True)
llk3_gs_ho_mc, df3_gs_ho_mc, full_params3_gs_ho_mc = build_mnlogit(design_x, y, debug='Y')
test_dev_gs_ho_mc = 2 * (llk3_gs_ho_mc - llk2_gs_ho)
test_df_gs_ho_mc = df3_gs_ho_mc - df2_gs_ho
test_p_value_gs_ho_mc = scipy.stats.chi2.sf(test_dev_gs_ho_mc, test_df_gs_ho_mc)

# intercept + group_size + homeowner + married_couple + group_size * homeowner
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
# create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)
llk4_gs_ho_mc_gsho, df4_gs_ho_mc_gsho, full_params4_gs_ho_mc_gsho = build_mnlogit(design_x, y, debug='Y')
test_dev_gs_ho_mc_gsho = 2 * (llk4_gs_ho_mc_gsho - llk3_gs_ho_mc)
test_df_gs_ho_mc_gsho = df4_gs_ho_mc_gsho - df3_gs_ho_mc
test_p_value_gs_ho_mc_gsho = scipy.stats.chi2.sf(test_dev_gs_ho_mc_gsho, test_df_gs_ho_mc_gsho)

print()
print("=" * 50)
print("ML-Assignment 4-Question 1-Section a)")
print("=" * 50)
# a)	(5 points) List the aliased parameters that you found in your model.
# intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * married_couple
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
# create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)
# create the columns for the homeowner * married_couple interaction effect
x_homc = create_interaction(x_ho, x_mc)
design_x = design_x.join(x_homc)
design_x = stats.add_constant(design_x, prepend=True)
llk5_gs_ho_mc_gsho_homc, df5_gs_ho_mc_gsho_homc, full_params5_gs_ho_mc_gsho_homc = build_mnlogit(design_x, y, debug='Y')
test_dev_gs_ho_mc_gsho_homc = 2 * (llk5_gs_ho_mc_gsho_homc - llk4_gs_ho_mc_gsho)
test_df_gs_ho_mc_gsho_homc = df5_gs_ho_mc_gsho_homc - df4_gs_ho_mc_gsho
test_p_value_gs_ho_mc_gsho_homc = scipy.stats.chi2.sf(test_dev_gs_ho_mc_gsho_homc, test_df_gs_ho_mc_gsho_homc)

print()
print("=" * 50)
print("ML-Assignment 4-Question 1-Section b)")
print("=" * 50)
# b)	(5 points) How many degrees of freedom do you have in your model?
print(f'Degree of Freedom = {df5_gs_ho_mc_gsho_homc}')

print()
print("=" * 50)
print("ML-Assignment 4-Question 1-Section c)")
print("=" * 50)
# c)	(10 points) After entering a model effect, calculate the Deviance test statistic, its degrees of freedom,
# and its significance value between the current model and the previous model.  List your Deviance test results by
# the model effects in a table.
print('Deviance Chi=Square Test')
print(f'==>for (Intercept + group_size) model')
print('Chi-Square Statistic = ', test_dev_gs)
print('  Degrees of Freedom = ', test_df_gs)
print('        Significance = ', test_p_value_gs)
print(f'==>for (Intercept + group_size + homeowner) model')
print('Chi-Square Statistic = ', test_dev_gs_ho)
print('  Degrees of Freedom = ', test_df_gs_ho)
print('        Significance = ', test_p_value_gs_ho)
print(f'==>for (Intercept + group_size + homeowner + married_couple) model')
print('Chi-Square Statistic = ', test_dev_gs_ho_mc)
print('  Degrees of Freedom = ', test_df_gs_ho_mc)
print('        Significance = ', test_p_value_gs_ho_mc)
print(f'==>for (Intercept + group_size + homeowner + married_couple + group_size * homeowner) model')
print('Chi-Square Statistic = ', test_dev_gs_ho_mc_gsho)
print('  Degrees of Freedom = ', test_df_gs_ho_mc_gsho)
print('        Significance = ', test_p_value_gs_ho_mc_gsho)
print(
    f'==>for (Intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * '
    f'married_couple) model')
print('Chi-Square Statistic = ', test_dev_gs_ho_mc_gsho_homc)
print('  Degrees of Freedom = ', test_df_gs_ho_mc_gsho_homc)
print('        Significance = ', test_p_value_gs_ho_mc_gsho_homc)

# apply chi-square and deviance test on given data
predictor = ['group_size', 'homeowner', 'married_couple']

test_result = pandas.DataFrame(index=predictor,
                               columns=['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])

for pred in predictor:
    deviance_stat, deviance_df, deviance_sig, mc_fadden_r_sq = deviance_test(purchase_likelihood[pred],
                                                                             purchase_likelihood['A'], debug='N')
    test_result.loc[pred] = ['Deviance', deviance_stat, deviance_df, deviance_sig, 'McFadden''s R^2', mc_fadden_r_sq]

print(test_result[test_result['Test'] == 'Deviance'])

print()
print("=" * 50)
print("ML-Assignment 4-Question 1-Section d)")
print("=" * 50)
# d)	(5 points) Calculate the Feature Importance Index as the negative base-10 logarithm of the significance value.
# List your indices by the model effects.
print(f'Feature Importance Index for (Intercept + group_size) \n= {-(numpy.log10(test_p_value_gs))}')
print(f'Feature Importance Index for (Intercept + group_size + homeowner) \n= {-(numpy.log10(test_p_value_gs_ho))}')
print(
    f'Feature Importance Index for (Intercept + group_size + homeowner + married_couple) \n'
    f'= {-(numpy.log10(test_p_value_gs_ho_mc))}')
print(
    f'Feature Importance Index for (Intercept + group_size + homeowner + married_couple + group_size * homeowner) \n'
    f'= {-(numpy.log10(test_p_value_gs_ho_mc_gsho))}')
print(
    f'Feature Importance Index for (Intercept + group_size + homeowner + married_couple + group_size * homeowner + '
    f'homeowner * married_couple) \n= {-(numpy.log10(test_p_value_gs_ho_mc_gsho_homc))}')

# create train data for train MNLogit model
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)
# Create the columns for the homeowner * married_couple interaction effect
x_homc = create_interaction(x_ho, x_mc)
design_x = design_x.join(x_homc)
design_x = stats.add_constant(design_x, prepend=True)

# train MNLogit model
logit = stats.MNLogit(y, design_x)
this_fit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)

print()
print("=" * 50)
print("ML-Assignment 4-Question 1-Section e)")
print("=" * 50)
# e)	(10 points) For each of the sixteen possible value combinations of the three features, calculate the predicted
# probabilities for A = 0, 1, 2 based on the multinomial logistic model.  List your answers in a table with proper
# labelling.
gs_d = [1, 2, 3, 4]
ho_d = [0, 1]
mc_d = [0, 1]
A_d = [0, 1, 2]
# create all possible combinations of features
x_data = []

for gsd in gs_d:
    for hod in ho_d:
        for mcd in mc_d:
            data = [gsd, hod, mcd]
            x_data = x_data + [data]

# create input data for prediction through the model
x_input = pandas.DataFrame(x_data, columns=['group_size', 'homeowner', 'married_couple'])
x_gs = pandas.get_dummies(x_input[['group_size']].astype('category'))
x_ho = pandas.get_dummies(x_input[['homeowner']].astype('category'))
x_mc = pandas.get_dummies(x_input[['married_couple']].astype('category'))
x_design = x_gs
x_design = x_design.join(x_ho)
x_design = x_design.join(x_mc)
# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
x_design = x_design.join(x_gsho)
x_design = stats.add_constant(x_design, prepend=True)
# Create the columns for the homeowner * married_couple interaction effect
x_homc = create_interaction(x_ho, x_mc)
x_design = x_design.join(x_homc)
x_design = stats.add_constant(x_design, prepend=True)
A_pred = this_fit.predict(exog=x_design)
# map the columns
A_pred['p_a_0'] = A_pred[0]
A_pred['p_a_1'] = A_pred[1]
A_pred['p_a_2'] = A_pred[2]

A_output = pandas.concat([x_input, A_pred[['p_a_0', 'p_a_1', 'p_a_2']]], axis=1)
print(A_output)

print()
print("=" * 50)
print("ML-Assignment 4-Question 1-Section f)")
print("=" * 50)
# f)	(5 points) Based on your model, what values of group_size, homeowner, and married_couple will maximize the
# odds value Prob(A=1) / Prob(A = 0)?  What is that maximum odd value?
A_output['odd value(p_a_1/p_a_0)'] = A_output['p_a_1'] / A_output['p_a_0']
print(A_output[['group_size', 'homeowner', 'married_couple', 'odd value(p_a_1/p_a_0)']])
print(A_output.loc[A_output['odd value(p_a_1/p_a_0)'].idxmax()])

print()
print("=" * 50)
print("ML-Assignment 4-Question 1-Section g)")
print("=" * 50)
# g)	(5 points) Based on your model, what is the odds ratio for group_size = 3 versus group_size = 1,
# and A = 2 versus A = 0?  Mathematically, the odds ratio is (Prob(A=2)/Prob(A=0) | group_size = 3) / ((Prob(
# A=2)/Prob(A=0) | group_size = 1). calculate odd for (Prob(A=2)/Prob(A=0) | group_size = 3)
pr_a_2_g_gs_3 = (purchase_likelihood[purchase_likelihood['group_size'] == 3].groupby('A').size()[2] /
                 purchase_likelihood[purchase_likelihood['group_size'] == 3].shape[0])
pr_a_0_g_gs_3 = (purchase_likelihood[purchase_likelihood['group_size'] == 3].groupby('A').size()[0] /
                 purchase_likelihood[purchase_likelihood['group_size'] == 3].shape[0])
o1 = pr_a_2_g_gs_3 / pr_a_0_g_gs_3
# calculate odd for (Prob(A=2)/Prob(A=0) | group_size = 1)
pr_a_2_g_gs_1 = (purchase_likelihood[purchase_likelihood['group_size'] == 1].groupby('A').size()[2] /
                 purchase_likelihood[purchase_likelihood['group_size'] == 1].shape[0])
pr_a_0_g_gs_1 = (purchase_likelihood[purchase_likelihood['group_size'] == 1].groupby('A').size()[0] /
                 purchase_likelihood[purchase_likelihood['group_size'] == 1].shape[0])
o2 = pr_a_2_g_gs_1 / pr_a_0_g_gs_1
# calculate odd ratio
o_r = o1 / o2
print(o_r)

print()
print("=" * 50)
print("ML-Assignment 4-Question 1-Section h)")
print("=" * 50)
# h)	(5 points) Based on your model, what is the odds ratio for homeowner = 1 versus homeowner = 0, and A = 0
# versus A = 1?  Mathematically, the odds ratio is (Prob(A=0)/Prob(A=1) | homeowner = 1) / ((Prob(A=0)/Prob(A=1) |
# homeowner = 0). calculate odd for (Prob(A=0)/Prob(A=1) | homeowner = 1)
pr_a_0_g_ho_1 = (purchase_likelihood[purchase_likelihood['homeowner'] == 1].groupby('A').size()[0] /
                 purchase_likelihood[purchase_likelihood['homeowner'] == 1].shape[0])
pr_a_1_g_ho_1 = (purchase_likelihood[purchase_likelihood['homeowner'] == 1].groupby('A').size()[1] /
                 purchase_likelihood[purchase_likelihood['homeowner'] == 1].shape[0])
o1 = pr_a_0_g_ho_1 / pr_a_1_g_ho_1
# calculate odd for (Prob(A=0)/Prob(A=1) | homeowner = 0)
pr_a_0_g_ho_0 = (purchase_likelihood[purchase_likelihood['homeowner'] == 0].groupby('A').size()[0] /
                 purchase_likelihood[purchase_likelihood['homeowner'] == 0].shape[0])
pr_a_1_g_ho_0 = (purchase_likelihood[purchase_likelihood['homeowner'] == 0].groupby('A').size()[1] /
                 purchase_likelihood[purchase_likelihood['homeowner'] == 0].shape[0])
o2 = pr_a_0_g_ho_0 / pr_a_1_g_ho_0
# calculate odd ratio
o_r = o1 / o2
print(o_r)
