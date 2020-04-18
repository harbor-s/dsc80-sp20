
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    
    labs = []
    projects = []
    discs = []
    checkpoints = []

    for column_name in grades.columns:
        if column_name[:3] == 'lab' and column_name[-2:].isnumeric():
            labs.append(column_name)
        elif column_name[:7] == 'project' and column_name[-2:].isnumeric():
            if column_name[10:20] == 'checkpoint':
                checkpoints.append(column_name)
            else:
                projects.append(column_name)
        elif column_name[:4] == 'disc' and column_name[-2:].isnumeric():
            discs.append(column_name)
        
    names = {
        'lab' : labs,
        'project': projects,
        'midterm': ['Midterm'],
        'final': ['Final'],
        'disc' : discs,
        'checkpoint' : checkpoints
    }
    
    return names


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    proj_names = get_assignment_names(grades)['project']
    proj_grades_sum = np.array([float(0) for _ in range(grades.shape[0])])
    grades = grades.fillna(0)
    
    for proj in proj_names:
        points = grades[proj]
        max_points = grades[f'{proj} - Max Points']        
        if f'{proj}_free_response' in grades.columns:
            points += grades[f'{proj}_free_response']
            max_points += grades[f'{proj}_free_response - Max Points']
        
        proj_grade = points/max_points
        proj_grades_sum += proj_grade
        
    proj_grades = proj_grades_sum/len(proj_names)
    return proj_grades


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """

    assignment_names = get_assignment_names(grades)
    
    false_late_count = []
    for l in assignment_names['lab']:
        lateness = f'{l} - Lateness (H:M:S)'
        false_late_count.append(grades[lateness][grades[lateness] < '07:00:00'][grades[lateness] > '00:00:00'].count())

    last_minutes = pd.Series(false_late_count, assignment_names['lab'])

    return last_minutes


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.8, 0.5}
    True
    """
        
    penalty = col.apply(lambda x: 1.0 if x < '07:00:00' else 0.9 if x < '168:00:00' else 0.8 if x < '336:00:00' else 0.5)
    
    return penalty


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """

    lab_names = get_assignment_names(grades)['lab']
    labdf = grades[lab_names]

    for l in lab_names:
        #print(labdf[l] / grades[f'{l} - Max Points'])
        #print(lateness_penalty(grades[f'{l} - Lateness (H:M:S)']))
        labdf.loc[:,l] = (labdf[l] * lateness_penalty(grades[f'{l} - Lateness (H:M:S)'])) / grades[f'{l} - Max Points']

    return labdf


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """

    tot = (processed.sum(axis=1) - processed.min(axis=1)) / (processed.shape[1] - 1)
    return tot


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    assgn_names = get_assignment_names(grades)

    proj_tot = projects_total(grades)
    lab_tot = lab_total(process_labs(grades)).fillna(0)

    checkpoint_names = assgn_names['checkpoint']
    checkpoint_grades_sum = [float(0) for _ in range(grades.shape[0])]
    for checkpoint in checkpoint_names:
        checkpoint_grade = grades[checkpoint] / grades[f'{checkpoint} - Max Points']
        checkpoint_grade = checkpoint_grade.fillna(0)
        checkpoint_grades_sum += checkpoint_grade
    checkpoint_tot = checkpoint_grades_sum / len(checkpoint_names)

    disc_names = assgn_names['disc']
    disc_grades_sum = [float(0) for _ in range(grades.shape[0])]
    for disc in disc_names:
        disc_grade = grades[disc] / grades[f'{disc} - Max Points']
        disc_grade = disc_grade.fillna(0)
        disc_grades_sum += disc_grade
    disc_tot = disc_grades_sum / len(disc_names)

    mid = grades['Midterm'] / grades['Midterm - Max Points']
    mid = mid.fillna(0)

    fin = grades['Final'] / grades['Final - Max Points']
    fin = fin.fillna(0)

    total = (0.20 * lab_tot) + (0.30 * proj_tot) + (0.025 * checkpoint_tot) + (0.025 * disc_tot) + (0.15 * mid) + (0.30 * fin)
    
    return total


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """

    letters = total.apply(lambda x: 'A' if x >= 0.9 else 'B' if x >= 0.8 else 'C' if x >= 0.7 else 'D' if x >= 0.6 else 'F')
    return letters


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """

    letter_grades = (final_grades(total_points(grades)))
    proportions = letter_grades.value_counts(normalize=True)
    return proportions

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of sophomores
    was no better on average than the class
    as a whole (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """

    soph_not_better = []
    
    for i in range (N):
        samp = grades.sample(frac = 1, replace = True)
        avg_all = total_points(samp).mean()
        avg_so = total_points(samp.loc[grades['Level'] == 'SO']).mean()

        if avg_so <= avg_all:
            soph_not_better.append(1)
        else:
            soph_not_better.append(0)
    p_val = np.count_nonzero(soph_not_better) / len(soph_not_better)
    return p_val

# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    assgn_names = get_assignment_names(grades)
    grades = grades.fillna(0)
    num_rows = grades.shape[0]

    proj_names = get_assignment_names(grades)['project']
    proj_grades_sum = np.array([float(0) for _ in range(grades.shape[0])])

    for proj in proj_names:
        points = grades[proj]
        max_points = grades[f'{proj} - Max Points']
        if f'{proj}_free_response' in grades.columns:
            points += grades[f'{proj}_free_response']
            max_points += grades[f'{proj}_free_response - Max Points']

        proj_grade = (points / max_points) + np.random.normal(0, 0.02, size=(num_rows))
        proj_grade = np.clip(proj_grade,0,1)
        proj_grades_sum += proj_grade

    proj_tot = proj_grades_sum / len(proj_names)

    lab_names = get_assignment_names(grades)['lab']
    labdf = grades[lab_names]

    for l in lab_names:
        labdf.loc[:, l] = np.clip(\
                                ((labdf[l] / grades[f'{l} - Max Points'])\
                                  + np.random.normal(0, 0.02, size=(num_rows)))\
                                * lateness_penalty(grades[f'{l} - Lateness (H:M:S)'])\
                        ,0,1)

    lab_tot = lab_total(labdf).fillna(0)


    checkpoint_names = assgn_names['checkpoint']
    checkpoint_grades_sum = [float(0) for _ in range(grades.shape[0])]
    for checkpoint in checkpoint_names:
        checkpoint_grade = grades[checkpoint] / grades[f'{checkpoint} - Max Points']
        checkpoint_grade = checkpoint_grade.fillna(0) + np.random.normal(0, 0.02, size=(num_rows))
        checkpoint_grade = np.clip(checkpoint_grade,0,1)
        checkpoint_grades_sum += checkpoint_grade
    checkpoint_tot = checkpoint_grades_sum / len(checkpoint_names)

    disc_names = assgn_names['disc']
    disc_grades_sum = [float(0) for _ in range(grades.shape[0])]
    for disc in disc_names:
        disc_grade = grades[disc] / grades[f'{disc} - Max Points']
        disc_grade = disc_grade.fillna(0) + np.random.normal(0, 0.02, size=(num_rows))
        disc_grade = np.clip(disc_grade,0,1)
        disc_grades_sum += disc_grade
    disc_tot = disc_grades_sum / len(disc_names)

    mid = grades['Midterm'] / grades['Midterm - Max Points']
    mid = mid.fillna(0) + np.random.normal(0, 0.02, size=(num_rows))
    mid = np.clip(mid,0,1)

    fin = grades['Final'] / grades['Final - Max Points']
    fin = fin.fillna(0) + np.random.normal(0, 0.02, size=(num_rows))
    fin = np.clip(fin,0,1)

    total = (0.20 * lab_tot) + (0.30 * proj_tot) + (0.025 * checkpoint_tot) + (0.025 * disc_tot) + (0.15 * mid) + (
                0.30 * fin)

    return total


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4], bool)
    True
    """

    return [0.0001, 0.85, [79,86], 0.06, True]

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
