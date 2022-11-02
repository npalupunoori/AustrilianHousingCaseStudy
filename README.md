# Austrilian housing case study
> A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. For the same purpose, the company has collected a data set from the sale of houses in Australia. The data is provided in the CSV file below.

>The company is looking at prospective properties to buy to enter the market. You are required to build a regression model using regularisation in order to predict the actual value of the prospective properties and decide whether to invest in them or not.

The company wants to know:
* Which variables are significant in predicting the price of a house, and
* How well those variables describe the price of a house.

## General Information
- Required to model the price of houses with the available independent variables. 
- This model will then be used by the management to understand how exactly the prices vary with the variables. 
- They can accordingly manipulate the strategy of the firm and concentrate on areas that will yield high returns. 
- Further, the model will be a good way for management to understand the pricing dynamics of a new market.

## Conclusions from Data analysis
- Masonry vaneer area has a positive effect on sale price
- Finished Basement, increase the cost of the House
- Higher living area means higher sales price
- As expected, Newer houses have higher Sale price.
- 'MSSubClass' it shows newer dwelling 20,60,120 has higher mean of sales price as well as higher max values.
- Number of 'GarageCars' at 3 has higher mean 'salePrice', For some reason 'GarageCars' with 4 has lower mean value.
- MSZoning in Low density and Floating village residental has high mean value for SalePrice, Low density has even more houses sold at higer pricess then any other zoning region Comerial zoning region has the lowest sale price as expected.
- Lot shape does not have much effect on the sale price
- Northridge, Northridge Heights, Stone Brook Neighborhood has high mean 'SalePrice' compared to other neighborhood's.
- Higher the dwellings has higher 'SalePrice'
- Brick FAce, Asbestos shingles, Cinder Block has negative effect on the sale price.
- Exterior quality has great effect on sale price, Excellent has Higher sale price. Faie has negative effect on sale price.
- Having basement with high cealings increased the cost of sale price tremendosuly. Noting having basement does not have high much effect instead of normall or low height.
- Good Living quarters have higher sales price
- As Expected Excelent heating quality and condition has higher Sale price
- Having a Excelent Kitchen quality has very positive effect on sale price
- Car Port has a negative effect on garage type. BuiltIn Gargae has increased the price of the house.

## Conclusions form regression model
* LotFrontage has negative effect on the housing price. Lot of coustomers are perfering not to have a big road connecting to there lot. This sits with the analysis before that say's Mean for homes in viallages and isolated neighbour hoods is high
* LotArea, TotalBsmtSF, GrLivArea,  is positive effect as expected
* Finished basement is increasing the house price
* House Age and Garage Age, Older houses tend to have lower sale price
* House price is effected by the dwelling of the house
* Neighborhood is having effect on sale price. There are neighbour hoods where people prefer to over others
    * Blueste , BrDale , CollgCr , Edwards , Gilbert , IDOTRR , MeadowV , Mitchel , NAmes , NPkVill , NWAmes , OldTown , SWISU , Sawyer People prefer these neighbourhood over others
    * BrkSid , ClearC , Crawfo , NoRidg , NridgH , Sawyer , Somers , StoneB , Timbe , Veenke are not prefering these neighbourhoods.
* Over all quality needs to be greater then 6 to have a positive effect on saleprice
* Over all condition needs to be atleast 5
* People are perfering 3 full bath over 1 and 2
* Coustomers what bedroom above ground
* Coustomers prefer to have atleast 9 rooms with 3 full bath
* Coustomers dont like to have more then one Full bath in basement
* Coustomers dont like carport
* Coustomers prefer with atlest 3 garage cars
* Remodeling has minimum effect on the sale price
## Contact
Created by [@npalupunoori] - feel free to contact me!


<!-- ## License -->
Copyright [2022] [Naren Reddy Palupunoori]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
