``` python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

bakery = pd.read_csv("downloads/bakery.csv")
bakery.head(50)

print(bakery.info())
```
### Original dataset 



<img src="images/info" width="500" height="300" />

#### Looking at the info() above, things to take note
- All the data fields are supposedly filled, no NA values
- 'DateTime' column needs to be converted from Object to Datetime
- There seems to be duplicate rows
- DayType seems to be incorrect e.g. dates that are Sunday are mislabelled as weekend (added this in after date manipulation futher below. Will rectify in earlier part of code)
-------------------
#### Convert "DateTime" column
- Converts the DateTime column to a datetime format for consistent processing.
- Creates a new column, DayType_new, to classify days as Weekday or Weekend based on the day of the week.
- Extracts and organizes date components:
  -  Month: Formatted as abbreviated names and ordered chronologically.
  - Date: Extracted as the specific calendar date.
  - month_year: Represents the month and year as a period for grouped analysis.
    
```python
bakery["DateTime"] = pd.to_datetime(bakery["DateTime"])

# added this code to address issue 4. above
bakery["DayType_new"] = np.where(bakery["DateTime"].dt.dayofweek < 5, 'Weekday', 'Weekend')

# add exrtact date, months and years from DateTime column
bakery["Month"] = bakery["DateTime"].dt.strftime("%b")
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
bakery["Month"] = pd.Categorical(bakery["Month"], categories=month_order, ordered=True)
bakery["Date"] = bakery["DateTime"].dt.date
bakery["month_year"] = bakery["DateTime"].dt.to_period("M")
bakery.info()

```
<img src="images/date_info" width="800" height="600" />

#### Inspect duplicate rows

- Using duplicated() and groupby() function, inspect the duplicate counts.
- If there are irregular counts of duplicates, I will assume that the data is **NOT** a duplicate, but rather an order for >1 of an item in a transaction (e.g. 2 Scandinavian in TransactionNo 2)

``` python
duplicate_rows = bakery[bakery.duplicated(keep=False)]
duplicate_rows['Duplicate Count'] = duplicate_rows.groupby(list(duplicate_rows.columns)).transform('size')

unique_duplicates = duplicate_rows['Duplicate Count'].unique()

if np.array_equal(unique_duplicates, [2]):
    print("Yes, the data most likely are showing up as duplicates")
else:
    print(f"There are values showing up in {', '.join(map(str, unique_duplicates))} rows.\nIt can be assumed that the duplicated rows are not duplicates but rather items in same transaction.")
```
------------
------------

### 1. What are the Top 5 most popular products and categories?
Solution:

1. Use value_counts() to get count of items, sort by descending Total count
2. Use iterrows(), get a string of top 5 from sorted data

``` python
# Count the occurrences of each unique item in the 'Items' column and reset the index to create a DataFrame
popular_item = bakery.value_counts("Items").reset_index().rename(columns = {"count":"No. purchased"})

# Rename the column "count" to "No. purchased" for better readability
popular_item_str = ', '.join(f"{row['Items']}" for _, row in popular_item.head(5).iterrows())

print(f"The Top 5 most popular items are: {popular_item_str}.")

```
