"""Streamlit app for Titanic survival analysis."""

import pandas as pd
import streamlit as st

from apputil import (survival_demographics, visualize_demographic,
                     family_groups, last_names, visualize_families,
                     determine_age_division, visualize_age_division)

# Load Titanic dataset
URL = ('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/'
       'data/titanic.csv')
df = pd.read_csv(URL)

st.write(
    '''
    # Titanic Visualization 1

    '''
)

# Display the research question (required by exercise)
research_question = (
    "**Research Question:** How do survival rates vary across different "
    "age groups and sexes within each passenger class? Do we see consistent "
    "gender-based survival advantages across all age categories and social "
    "classes?"
)
st.write(research_question)

# Display the survival demographics table (good practice)
st.write("## Survival Demographics Analysis")
demographics_data = survival_demographics()
st.dataframe(demographics_data)

# Generate and display the figure
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width=True)

st.write(
    '''
    # Titanic Visualization 2
    '''
)

# Display the research question for Exercise 2
family_research_question = (
    "**Research Question:** How does family size correlate with ticket "
    "fares across different passenger classes? Do larger families pay "
    "proportionally more or less per person, and does this pattern vary "
    "by social class?"
)
st.write(family_research_question)

# Display the family groups analysis
st.write("## Family Size and Wealth Analysis")
family_data = family_groups()
st.dataframe(family_data)

# Display last names analysis
st.write("## Last Names Analysis")
name_counts = last_names()
st.write("**Findings about last names vs family size data:**")

# Analyze the relationship between last name counts and family data
families_with_multiple_members = name_counts[name_counts > 1]
st.write(f"- **{len(families_with_multiple_members)}** different family names "
         f"appear multiple times in the passenger list")
st.write(f"- The largest family group is **{name_counts.max()}** people "
         f"with the surname '{name_counts.idxmax()}'")
st.write(f"- **{len(name_counts)}** total unique last names among "
         f"{name_counts.sum()} passengers")

st.write("This confirms that our family_size calculation captures real "
         "family relationships - passengers with the same last name "
         "likely traveled together as families.")

# Generate and display the families visualization
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width=True)

st.write(
    '''
    # Titanic Visualization Bonus
    '''
)

# Display bonus analysis description
st.write(
    "**Bonus Analysis:** This section explores whether being older or younger "
    "than the median age within your passenger class affected survival chances. "
    "Each passenger class has different age demographics, so we compare survival "
    "rates relative to the median age of their specific class."
)

# Show age division analysis
st.write("## Age Division Analysis")
age_division_data = determine_age_division()

# Display median ages for each class
median_ages = age_division_data.groupby('Pclass')['Age'].median()
st.write("**Median ages by passenger class:**")
for pclass, median_age in median_ages.items():
    st.write(f"- Class {pclass}: {median_age:.1f} years")

# Show sample of the data with new column
st.write("### Sample of data with older_passenger column:")
sample_data = age_division_data[['PassengerId', 'Name', 'Pclass', 'Age', 
                                'older_passenger', 'Survived']].head(10)
st.dataframe(sample_data)

# Generate and display the bonus visualization
fig3 = visualize_age_division()
st.plotly_chart(fig3, use_container_width=True)

# Additional insights
st.write("### Key Insights:")
survival_by_age_division = age_division_data.groupby(['Pclass', 'older_passenger'])['Survived'].mean()
st.write(
    "The visualization shows how survival rates varied between passengers "
    "who were above vs below the median age within their respective classes. "
    "This helps us understand if age advantages were consistent across "
    "social classes or if class privilege could overcome age disadvantages."
)