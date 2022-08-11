# Analysis of the Effects of Regulation on Railroad Safety

Kai Liao, Katrina Siegfried, Andrew Smith

This is the group project repository for Group 17 for CSCI5502 Data Mining at CU Boulder in Summer 2022.
This project uses the data mining techniques of FP Growth, Decision Trees, and K-Means Clustering to analyze highway-railroad crash data, the effects of previous regulations on crash risk, and key areas for future regulations to reduce crash risk.

Dataset Used: https://catalog.data.gov/dataset/highway-rail-grade-crossing-accident-data
Dataset Description: https://data.transportation.gov/Railroads/Highway-Rail-Grade-Crossing-Accident-Data/7wn6-i5b9

Questions Sought and Answers:

Q: What crossing characteristics are most central in highway-rail collisions, and what changes to these characteristics could reduce risk?
A: According to decision tree analysis, the speed of the train is a major determinant of casualty rate. Trains moving faster than 60 mph at intersections are more likely to cause 3+ injuries.

Q: What are frequent occurrences with highway-rail collisions, and are there any regulations that are ineffective or need improvement?
A: According to frequent pattern analysis, more safety signaling may be required - especially to prevent head-on and track obstacle accidents.

Q: How does location-specific factors, such as climate, affect the risk of highway-rail collisions? What can be done to mitigate these factors?
A: According to k-means clustering analysis, inclement weather crashes were more frequent than clear weather crashes in all locations, but snow was more common in the northern US while rain was more common in the southern US. Work may need to be done to increase visibility in these circumstances.

Applications:

In general, the analyses indicate that better signaling and speed control are required to reduce the risk of highway-railroad accidents.
Decision tree and k-means analysis indicated that speed and wet inclement weather were related to more frequent crashes, indicating that speed control could reduce risk - especially in cold slippery weather.
Frequent pattern and k-means analysis indicated that signaling reduced crash risk and that low-visibility weather correlated to more crashes, indicating that clearer signals could reduce risk.
Decision tree and frequent pattern analysis indicated that overall better design of highway-rail intersections could reduce crash risk, as speed limits and signaling implementations were not consistent across incidents.

Final Presentation:

Final Paper: