"""Utility functions for Titanic survival analysis."""

import pandas as pd
import plotly.express as px


# Exercise 1: Survival Patterns
def survival_demographics():
    """
    Analyze survival patterns on the Titanic by passenger class, sex, and age.

    Returns:
        pandas.DataFrame: A table with survival statistics for all combinations
                         of class, sex, and age group.
    """
    # Load Titanic dataset
    url = ('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/'
           'data/titanic.csv')
    df = pd.read_csv(url)
    
    # Create age categories using pd.cut
    age_bins = [0, 12, 19, 59, float('inf')]
    age_labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels,
                             right=True)
    
    # Create a complete index of all possible combinations
    from itertools import product
    all_classes = sorted(df['Pclass'].unique())
    all_sexes = sorted(df['Sex'].unique())
    all_age_groups = age_labels
    
    # Create MultiIndex with all combinations
    all_combinations = list(product(all_classes, all_sexes, all_age_groups))
    complete_index = pd.MultiIndex.from_tuples(
        all_combinations, 
        names=['Pclass', 'Sex', 'age_group']
    )
    
    # Group by class, sex, and age group (without observed=True to include empty groups)
    grouped = df.groupby(['Pclass', 'Sex', 'age_group'], observed=False)
    
    # Calculate survival statistics
    survival_stats = grouped.agg({
        'PassengerId': 'count',  # Total passengers
        'Survived': ['sum', 'mean']  # Survivors and survival rate
    }).round(3)
    
    # Reindex to include all combinations (fills missing with 0)
    survival_stats = survival_stats.reindex(complete_index, fill_value=0)
    
    # Flatten column names
    survival_stats.columns = ['n_passengers', 'n_survivors', 'survival_rate']
    
    # Fix survival rate for groups with 0 passengers (set to 0.0 instead of NaN)
    survival_stats.loc[survival_stats['n_passengers'] == 0, 'survival_rate'] = 0.0
    
    # Reset index to make grouping columns regular columns
    survival_stats = survival_stats.reset_index()
    
    # Sort for easy interpretation (by class, then sex, then age group)
    survival_stats = survival_stats.sort_values(['Pclass', 'Sex',
                                                  'age_group'])
    
    # Reset index for clean display
    survival_stats = survival_stats.reset_index(drop=True)
    
    return survival_stats


def visualize_demographic():
    """
    Create a Plotly visualization to explore survival patterns by demographics.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure showing survival rates
                                    by age group and sex across passenger
                                    classes.
    """
    # Get the survival demographics data
    data = survival_demographics()

    # Create a grouped bar chart showing survival rates by age group and sex,
    # with separate subplots for each passenger class
    fig = px.bar(
        data,
        x='age_group',
        y='survival_rate',
        color='Sex',
        facet_col='Pclass',
        title='Survival Rates by Age Group, Sex, and Passenger Class',
        labels={
            'survival_rate': 'Survival Rate',
            'age_group': 'Age Group',
            'Pclass': 'Passenger Class'
        },
        text='survival_rate',
        color_discrete_map={'male': '#2E86AB', 'female': '#A23B72'}
    )

    # Customize the layout
    fig.update_layout(
        showlegend=True,
        font_size=12,
        title_font_size=16,
        height=500
    )

    # Format text on bars to show percentages
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')

    # Update y-axis to show percentages
    fig.update_yaxes(tickformat='.0%', range=[0, 1.1])

    # Update facet titles
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.replace("Pclass=", "Class "))
    )

    return fig


# Exercise 2: Family Size and Wealth
def family_groups():
    """
    Explore the relationship between family size, passenger class, and fare.

    Returns:
        pandas.DataFrame: A table with family size and fare statistics
                         grouped by family size and passenger class.
    """
    # Load Titanic dataset
    url = ('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/'
           'data/titanic.csv')
    df = pd.read_csv(url)
    
    # Create family_size column: SibSp + Parch + 1 (the passenger themselves)
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    
    # Group by family size and passenger class
    grouped = df.groupby(['family_size', 'Pclass'])
    
    # Calculate statistics for each group
    family_stats = grouped.agg({
        'PassengerId': 'count',  # Total passengers
        'Fare': ['mean', 'min', 'max']  # Fare statistics
    }).round(2)
    
    # Flatten column names
    family_stats.columns = ['n_passengers', 'avg_fare', 'min_fare', 'max_fare']
    
    # Reset index to make grouping columns regular columns
    family_stats = family_stats.reset_index()
    
    # Sort by class first, then by family size for easy interpretation
    family_stats = family_stats.sort_values(['Pclass', 'family_size'])
    
    # Reset index for clean display
    family_stats = family_stats.reset_index(drop=True)
    
    return family_stats


def last_names():
    """
    Extract last names from passenger names and count occurrences.

    Returns:
        pandas.Series: Count of each last name (last name as index).
    """
    # Load Titanic dataset
    url = ('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/'
           'data/titanic.csv')
    df = pd.read_csv(url)
    
    # Extract last names (text before the first comma)
    # Names are in format: "Last, Title. First"
    last_names_series = df['Name'].str.split(',').str[0].str.strip()
    
    # Count occurrences of each last name
    name_counts = last_names_series.value_counts()
    
    return name_counts


def visualize_families():
    """
    Create a Plotly visualization exploring family size and wealth patterns.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure showing the relationship
                                    between family size, passenger class,
                                    and ticket fares.
    """
    # Get the family groups data
    data = family_groups()
    
    # Create a scatter plot showing average fare vs family size,
    # with size representing number of passengers
    fig = px.scatter(
        data,
        x='family_size',
        y='avg_fare',
        color='Pclass',
        size='n_passengers',
        title='Family Size vs Average Fare by Passenger Class',
        labels={
            'family_size': 'Family Size',
            'avg_fare': 'Average Fare (£)',
            'Pclass': 'Passenger Class',
            'n_passengers': 'Number of Passengers'
        },
        color_discrete_map={1: '#2C3E50', 2: '#E74C3C', 3: '#BDC3C7'},
        hover_data=['min_fare', 'max_fare']
    )
    
    # Customize the layout
    fig.update_layout(
        showlegend=True,
        font_size=12,
        title_font_size=16,
        height=500,
        legend_title_text='Passenger Class'
    )
    
    # Update x-axis to show all family sizes
    fig.update_xaxes(title='Family Size', dtick=1)
    
    # Update y-axis for better fare display
    fig.update_yaxes(title='Average Fare (£)')
    
    # Add trend lines for each class
    fig.update_traces(
        mode='markers+lines',
        line=dict(width=1.5),
        marker=dict(line=dict(width=0.5, color='white'), opacity=0.8)
    )
    
    return fig


# Bonus Question: Age Division Analysis
def determine_age_division():
    """
    Add older_passenger column based on median age for each passenger class.

    Returns:
        pandas.DataFrame: Updated Titanic dataset with older_passenger column
                         (True if above median age for their class, False otherwise).
    """
    # Load Titanic dataset
    url = ('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/'
           'data/titanic.csv')
    df = pd.read_csv(url)
    
    # Calculate median age for each passenger class (excluding NaN values)
    median_ages = df.groupby('Pclass')['Age'].median()
    
    # Create older_passenger column using map and comparison
    df['class_median_age'] = df['Pclass'].map(median_ages)
    df['older_passenger'] = df['Age'] > df['class_median_age']
    
    # Handle NaN values in Age column - set to False for missing ages
    df['older_passenger'] = df['older_passenger'].fillna(False)
    
    # Drop the temporary column
    df = df.drop('class_median_age', axis=1)
    
    return df


def visualize_age_division():
    """
    Visualize age division patterns and their relation to survival.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure showing age division
                                    analysis across passenger classes.
    """
    # Get the data with age division
    df = determine_age_division()
    
    # Create survival analysis by age division and class
    age_survival = df.groupby(['Pclass', 'older_passenger']).agg({
        'PassengerId': 'count',
        'Survived': ['sum', 'mean']
    }).round(3)
    
    # Flatten column names
    age_survival.columns = ['n_passengers', 'n_survivors', 'survival_rate']
    age_survival = age_survival.reset_index()
    
    # Create labels for age division
    age_survival['age_division'] = age_survival['older_passenger'].map({
        True: 'Above Median Age',
        False: 'Below/At Median Age'
    })
    
    # Create grouped bar chart
    fig = px.bar(
        age_survival,
        x='Pclass',
        y='survival_rate',
        color='age_division',
        title='Survival Rates by Age Division Within Each Passenger Class',
        labels={
            'survival_rate': 'Survival Rate',
            'Pclass': 'Passenger Class',
            'age_division': 'Age Relative to Class Median'
        },
        text='survival_rate',
        color_discrete_map={
            'Above Median Age': '#34495E',
            'Below/At Median Age': '#95A5A6'
        },
        barmode='group'
    )
    
    # Customize the layout
    fig.update_layout(
        showlegend=True,
        font_size=12,
        title_font_size=16,
        height=500,
        xaxis_title='Passenger Class'
    )
    
    # Format text on bars to show percentages
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    
    # Update y-axis to show percentages
    fig.update_yaxes(tickformat='.0%', range=[0, 1.1])
    
    return fig


# update/add code below ...