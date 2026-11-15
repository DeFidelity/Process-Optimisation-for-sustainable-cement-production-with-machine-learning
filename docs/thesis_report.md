## **PROCESS OPTIMIZATION FOR SUSTAINABLE CEMENT PRODUCTION USING MACHINE LEARNING** 

## **BY** 

## **AMUSAT HAKI ADEYEMI 18/182145001** 

## **SUBMITTED TO:** 

## **DEPARTMENT OF CHEMICAL/PETROCHEMICAL ENGINEERING FACULTY OF ENGINEERING AND TECHNOLOGY UNIVERSITY OF CALABAR CALABAR** 

## **IN PARTIAL FULFILLMENT FOR THE AWARD OF BACHELOR OF ENGINEERING DEGREE (B.ENG) IN CHEMICAL ENGINEERING** 

**FEBRUARY, 2025** 

1 

## **CERTIFICATION** 

This is to certify that I, **Amusat, Haki Adeyemi,** with Matriculation Number: **18/182145001** 

carried out this project titled: **Process optimisation for sustainable cement production,** 

**using machine learning** under supervision and guidance of **Engr. Prof. Odey Osha** , department of Chemical engineering, faculty of engineering, university of Calabar, Calabar. 

_______________________ _________________ **Engr. Prof. Odey Osha Date** _Supervisor_ _______________________ _________________ **Engr. Dr. Ojong E. Ojong Date** _Head of Department_ _______________________ _________________ **Engr. Prof. Odey Osha Date** _Dean_ _______________________ _________________ **External Supervisor Date** 

2 

## **DEDICATION** 

This novelty body of work is dedicated to Almighty Allah, for fulfilling one of my prime childhood dream of becoming a chemical engineer against all odds poised agains it in the most wonderful way, ALHAMDULILAH. 

3 

## **ACKNOWLEDGEMENTS** 

Alhamdulillah, I want to give thanks to God Almighty for all the blessings in my life and for His unwavering support, giving me everything I ever ask for. Ya Allah, I know this is Your blessing, and I am not taking any part of it for granted. 

To my father, thank you for finally allowing me to pursue a degree in Engineering, even though it cost me a year to convince you that I wanted this over medicine. I’m very grateful for your support and approval. And to my mom, the most wonderful woman anyone could ever ask for. 

To the Dean of the Faculty of Engineering, Engr. Prof. Odey Osha, and the Head of Department, Engr. Dr. Ojong E. Ojong, thank you for accepting my project proposal. Even though it is novel, complex, and more inclined toward technology than it is to traditional chemical engineering, this shows how ready the department is for a technology-driven world that is already upon us. 

I will also like to acknowledge my family, the Abioduns, and the MSSN, especially my brother and my five-year roommate, Ridwanulahi Aderounmu. You’re amazing, and I’m happy that our paths crossed. 

This cannot be complete without thanking the good people of Lafarge Holcim for accepting me for an internship opportunity, which gave birth to this work, and for providing me with data to make the work a reality, especially Itunu Olatunji and Azeez. You guys are the best supervisors anyone could ask for. 

Finally, to all my friends in Lafarge, Yahuzah, Abubakar, and Adam especially; my coursemates, friends, and the MSSN, all thanks to you all. 

4 

## **ABSTRACT** 

The study on Process Optimization for sustainable cement production using machine learning has been achieved, the project considered the application of Random Forest Regressor and NSGA-II to predict and optimize energy consumption, CO2 emission and clinker production rate, error analysis models(% deviation and root mean squareerrors) where utilized to validate the models, the result from the simulation processat conditions 1168.21 kwh/t-clincker, 1005.78 kg/hr of energy consumption and CO2 where obtained atavery high R<sup>2</sup> values of 0.98. The optimal values obtained where conditions 1030.42 kwh/t-clincker, 920.42 kg/hr and 92.05 tph for energy consumption, CO2 emission and clincker production rate respectively. Negligible percentage deviation and RMSE values of 0.63%, 3.32%, and 0.00%, and 0.0734, 0.0323, indicates the high precision and sustainability of the models applied to solve indicators for optimal and sustainable cement production. 

|**TABLE OF CONTENT**||
|---|---|
|TITLE PAGE|i|
|CERTIFICATION|ii|
|DEDICATION|iii|
|ACKNOWLEDGEMENTS|vi|
|ABSTRACT|v|
|TABLE OF CONTENT|vi|
|**CHAPTER ONE: INTRODUCTION**|**1**|
|1.1 Background of the Study|1|
|1.2 Statement of Problem|2|
|1.3 Aim and Objectives|3|
|1.4 Expected Benefits|3|
|**CHAPTER TWO: LITERATURE REVIEW**|**6**|
|2.1. Reiterating the Research Problem: Balancing Efficiency and Sustainability in Ce|ment|
|Production|6|
|2.1. Cement Production Processes|7|
|2.1.1 Traditional Cement Production|7|
|2.1.2 Clinkerization Process and Environmental Impact|9|
|2.1.3  Quantification of the Environmental Impact of Cement Production|10|
|2.1.4 Current Sustainability Challenges in the Cement Industry|12|
|2.2 Sustainable Practices in Cement Production|13|
|2.2.1 Overview of sustainable practices and innovations in the cement industry:|13|
|2.2.2 Review of existing measures and technologies aimed at reducing the environm|ental|
|impact of the cement production process|13|
|2.2.3 Case studies of successful implementations of sustainable practices.|13|
|2.3. Process Optimization in Cement Manufacturing|14|
|2.3.1 Importance of process optimization in achieving sustainability:|14|
|2.4 Machine Learning in Industrial Process Optimization|15|



|2.4.1 Overview of machine learning and its relevance to process optimization:|16|
|---|---|
|2.4.2 Advantages of using machine learning for process optimization|17|
|2.5  Applications of Machine Learning in Cement Production|17|
|2.5.1 Review of existing studies that have applied machine learning in cement prod|uction 17|
|2.5.2 Predictive Maintenance and Optimization in Cement Plants|19|
|2.5.3 Development of Machine Learning Models for Predictive Maintenance|20|
|2.6 Challenges and Limitations of Machine Learning in Cement Production|21|
|2.6.1 Technical challenges in implementing machine learning|21|
|2.7. Synthesis of Literature and Research Gaps:|21|
|**CHAPTER THREE: MATERIALS AND METHOD**|**22**|
|3.1 Materials|22|
|3.1.1 Laptop|22|
|3.1.2 Plant dataset|22|
|3.2 Method|24|
|3.2.1Application of M L Models|24|
|3.2.2Multi Objectives Optimization Models Application|25|
|3.2.3 Model simulation and validation|28|
|3.3 Input Parameters|29|
|3.4 Solution Techniques|31|
|**CHAPTER FOUR: RESULTS**|**36**|
|4.1 Results|36|
|4.1.1 Results from Prediction Models|36|
|4.2 Optimization of the parameters|40|
|4.2.1 Pareto front Interpretation:|41|



|4.2.2 Optimal Values|41|
|---|---|
|4.3 Validation of Models|45|
|**CHAPTER FIVE: CONCLUSION AND RECOMMENDATIONS**|**47**|
|5.1 Conclusion|47|
|5.2 Recommendations|47|
|REFERENCES|49|



### **LIST OF TABLES** 

|Table 3.1:  Input Parameters|30|
|---|---|
|Table 4.1: Representative Pareto-Optimal Solutions|44|
|Table 4.2: Model Validation and Deviation Analysis|46|



## **LIST OF FIGURES** 

Figure 3.1: Image showing the rotary kiln flow diagram 

23 

|Figure 3.2: Code Structure and Algorithm|35|
|---|---|
|Figure 4.1: R² value chart for both the energy consumption and CO2emission chart.|37|
|Figure 4.2: Feature importance for the energy consumption prediction model.||
|The height of each bar represents the relative importance of the corresponding feature.|38|
|Figure 4.3: Feature importance for the CO2emissions prediction model. The bar heights<br>represent the relative importance of each feature.|39|
|Figure 4.4: Effect of energy consumption rate on CO2emission rate|40|



## **CHAPTER ONE** 

## **INTRODUCTION** 

## **1.1 Background of the Study** 

Cement is the fundamental building block of modern civilization.  Its applications extend far beyond construction, playing a vital role in various industries, Cement stands as the cornerstone of infrastructure development, serving as the binding agent in concrete, mortar, and grout. Its versatility and durability make it indispensable in various industries, including construction, transportation, and agriculture. From towering skyscrapers to intricate road networks, cement plays a vital role in shaping the modern world. 

However, traditional cement production comes at a significant environmental cost. Two major concerns stand out: 

CO2 Emissions is the process of producing cement, particularly during clinkerization (conversion of limestone to clinker), releases a substantial amount of carbon dioxide (CO2) into the atmosphere. Estimates suggest that the cement industry contributes roughly 8% of global CO2 emissions [Global CO2 emissions from cement production, 1928–2018 - ESSD Copernicus essd.copernicus.org]. This CO2 emission significantly contributes to climate change and its associated environmental issues. 

Cement production is an energy-intensive process, requiring high temperatures for clinkerization, the energy-intensive nature of cement manufacturing contributes to high energy consumption. This reliance on fossil fuels like coal further exacerbates the environmental impact by contributing to greenhouse gas emissions and air pollution [IPCC Sixth Assessment Report Climate Change 2021: Cambridge University Press. In press]. 

## **The Need for Sustainable Cement Production Practices:** 

The environmental impact of traditional cement production necessitates a shift towards sustainable practices. As the global population and infrastructure demand continue to grow, finding ways to reduce the environmental footprint of cement production becomes increasingly crucial.  Sustainable practices aim to: 

- i. Minimize CO2 emissions throughout the production process. 

- ii. Reduce reliance on fossil fuels and explore alternative energy sources. 

- iii. Utilize recycled materials and byproducts from other industries. 

- iv. Develop innovative production methods that promote energy efficiency. 

In the face of escalating environmental concerns, there is a growing imperative to transition 

to sustainable cement production. Sustainable practices aim to minimize the environmental impact of cement manufacturing while meeting the increasing demand for construction materials. 

By transitioning to sustainable cement production practices, the industry can contribute significantly to mitigating climate change and ensuring a cleaner environment for future generations. 

## **1.2 Statement of Problem** 

The cement industry faces the critical challenge of CO2 emission while maintaining or increasing clinker production rates. Traditional optimization methods often struggles to balance these competing objectives, necessitating a data-driven approach capable of identifying operating conditions that satisfies  all three criteria **.** 

## **1.3 Aim and Objectives** 

## **Aim** 

To develop a machine learning-driven optimization framework for sustainable cement production that can balance energy efficiency, CO2  emissions reduction, and clinker production rate. 

**Objectives** : 

1. Develop machine learning models to accurately predict energy consumption and CO2 emissions in cement production processes. 

2. Employ a multi-objective optimization algorithm (NSGA-II) to generate optimal solutions that balance energy efficiency, CO2 emissions reduction, and clinker production rate. 

3. Validate the effectiveness of the optimization framework using real-world data and case studies. 

## **1.5 Expected Benefits** 

The successful implementation of machine learning (ML) models in cement production processes holds tremendous potential for advancing sustainability objectives and optimizing operational efficiency on a global scale (Andrew, 2018; Chen _et al.,_ 2019). 

A well-designed ML model for real-time optimization in clinker production can revolutionize the cement industry, paving the way for a more sustainable future. The expected benefits of this research include: 

## i. **Reduced Energy Consumption:** 

By predicting and preventing deviations from optimal energy use, the ML model can significantly reduce energy consumption during clinker production (Scrivener _et al.,_ 2018). Lower energy usage translates into a reduced reliance on fossil fuels, which in turn mitigates greenhouse gas emissions and air pollution (Porter & van der Linde, 

1995). Consequently, the overall carbon footprint of the cement industry is decreased, contributing meaningfully to the global fight against climate change (Andrew, 2018). 

## ii. **Minimized CO** ₂ **Emissions:** 

Clinkerization is one of the major contributors to CO₂ emissions in cement production. Optimizing energy usage through ML-based process adjustments directly leads to fewer CO₂ emissions per unit of clinker produced (Chen _et al.,_ 2019). This reduction positively impacts global air quality and further supports initiatives to mitigate climate change (Zhang _et al.,_ 2020). 

## iii. **Enhanced Sustainability Practices:** 

ML-driven optimization enables cement plants to adopt more sustainable practices without compromising operational efficiency or productivity. By continuously monitoring and analyzing process parameters, ML models can identify inefficiencies and recommend targeted interventions to enhance resource efficiency, minimize environmental impact, and optimize overall plant performance (Kumar _et al.,_ 2018). This proactive, data-driven approach ensures that cement production remains economically viable while meeting stringent environmental regulations and industry standards (Porter & van der Linde, 1995). 

In summary, the integration of machine learning technologies in cement production not only yields tangible benefits—such as reduced energy consumption and lower carbon emissions—but also fosters the adoption of sustainable practices on a global scale. By harnessing the predictive capabilities of ML models, cement plants can achieve an optimal balance between operational efficiency and environmental responsibility, paving the way for a more sustainable future for both the industry and the planet (Chen _et al.,_ 2019; Porter & van der Linde, 1995). 

## **CHAPTER TWO** 

## **LITERATURE REVIEW** 

## **2.1. Reiterating the Research Problem: Balancing Efficiency and Sustainability in** 

## **Cement Production** 

The cement industry, a cornerstone of global infrastructure development, grapples with the formidable challenge of balancing production efficiency with environmental sustainability. This delicate equilibrium is essential due to the industry's significant environmental footprint, primarily stemming from its energy-intensive processes and substantial carbon dioxide (CO2) emissions. Cement production accounts for approximately 8% of global anthropogenic CO2 emissions (Andrew, 2018).]. 

While effective in meeting the escalating demand for construction materials, traditional cement production methods have raised substantial environmental concerns. The clinkerization process, a pivotal stage in cement manufacturing, is particularly energy-intensive and releases a considerable amount of CO2. This process' reliance on fossil fuels, such as coal, further exacerbates its environmental impact, contributing to greenhouse gas emissions and air pollution (Worrell _et al.,_ 2001). These emissions contribute significantly to climate change and necessitate a shift towards more sustainable practices. 

The pursuit of sustainable cement production entails minimizing CO2 emissions, reducing reliance on fossil fuels, and exploring alternative energy sources (Habert and Miller, 2012). It also involves utilizing recycled materials and byproducts from other industries, such as fly ash and slag, as supplementary cementitious materials (SCMs), and developing innovative production methods that prioritize energy efficiency. 

Despite these advancements, there remains a substantial gap in effectively integrating these strategies into a cohesive, optimized production process. Existing research primarily focuses on isolated aspects of efficiency or sustainability, without adequately addressing the interplay 

between these objectives. This research aims to fill this gap by leveraging machine learning techniques to develop a comprehensive process optimisation model that balances efficiency with sustainability in cement production. 

Optimizing for efficiency alone may neglect environmental consequences while prioritizing sustainability could potentially impact production speed and cost. By addressing this critical research problem, this study not only contributes to the academic discourse on sustainable industrial practices but also provides practical insights for the cement industry. The findings will have significant implications for policy-making, industry practices, and future research, promoting a more sustainable approach to cement production that aligns with global environmental goals. 

## **2.1. Review of Related Literatures** 

## **2.1.1 Traditional Cement Production** 

The traditional cement production process, also known as the dry process, involves several key steps: raw material extraction, raw material preparation, clinkerization, and cement grinding. Each of these stages contributes to the overall efficiency and environmental impact of the cement manufacturing process. 

1. **Raw Material Extraction and Preparation** : The primary raw materials for cement production are limestone, which provides calcium, and clay or shale, which contributes to silica, alumina, and iron oxide (Habert & Miller, 2012). These materials are quarried or mined and then transported to the cement plant. At the plant, the raw materials undergo crushing and grinding to reduce their size and ensure a homogeneous mixture (Habert & Miller, 2012). This preparation stage is crucial for achieving the desired chemical composition and reactivity of the raw mix. 

2. **Clinkerization** : The prepared raw mix is fed into a rotary kiln, a large rotating furnace. Inside the kiln, the raw mix is heated to temperatures exceeding 1450°C 

(2642°F) (Habert & Miller, 2012). This high-temperature process, known as clinkerization, triggers a series of chemical reactions that transform the raw materials into clinker. Clinker is a nodular material consisting of calcium silicates, aluminates, and ferrites, which are the primary constituents of cement (Habert & Miller, 2012). 

3. **Cement Grinding** : The clinker is cooled and then ground into a fine powder. During this stage, a small amount of gypsum (calcium sulfate) is added to regulate the setting time of the cement (Habert & Miller, 2012). The grinding process is essential for achieving the desired fineness of the cement, which affects its workability, strength development, and durability. The final product, Portland cement, is then stored in silos or packaged for distribution. 

The traditional cement production process is known for its significant environmental impacts, particularly in terms of energy consumption and CO₂ emissions. Approximately 60% of the total CO₂ emissions in cement production come from the chemical transformation of limestone (calcination), while the remaining 40% result from fuel combustion to heat the kiln (International Energy Agency, 2020). The high energy requirements and greenhouse gas emissions associated with clinkerization make it a focal point for efforts to improve the sustainability of cement production (Barker _et al.,_ 2021). 

## **2.1.2 Clinkerization Process and Environmental Impact** 

The clinkerization process is the most energy-intensive and environmentally impactful stage in cement production. It involves heating the raw material mixture to temperatures exceeding 1450°C (2642°F) in a rotary kiln (Habert & Miller, 2012). This high-temperature process triggers a series of chemical reactions, including the calcination of limestone (CaCO3) to 

release carbon dioxide (CO2) and the formation of calcium silicates, which are the primary strength-giving components of cement (Andrew, 2018). 

1. **CO** 2 **Emissions** : Clinkerization contributes substantially to CO2 emissions due to the calcination of limestone and the combustion of fossil fuels in the kiln. According to a study by Smith _et al_ . (2018), the clinker production process accounts for approximately 70-80% of the total CO2 emissions from cement manufacturing. During calcination, limestone (calcium carbonate) is heated in the kiln to produce lime (calcium oxide) and CO2. 

For every ton of clinker produced, approximately 0.8-0.9 tons of CO2 are emitted (Andrew, 2018). This makes the cement industry one of the largest industrial sources of CO2 emissions globally. In addition to calcination, the use of fossil fuels, such as coal and petroleum coke, as energy sources in the kiln also produces CO2 emissions. The combustion of these fuels releases CO2 that was previously sequestered underground, further exacerbating the environmental impact. 

2. **Energy Consumption:** The high temperatures required for clinkerization necessitate a substantial energy input, primarily derived from the combustion of fossil fuels such as coal, petroleum coke, and natural gas (Worrell _et al.,_ 2001). This reliance on fossil fuels not only contributes to greenhouse gas emissions but also exposes the industry to fluctuations in fuel prices and supply disruptions. The energy intensity of clinkerization varies depending on the kiln technology, fuel type, and raw material composition, but it typically ranges from 3-4 GJ/ton of clinker (Worrell _et al.,_ 2001). 

Efforts to reduce energy consumption in clinker production have focused on improving kiln efficiency and exploring alternative fuel sources. Advanced kiln technologies, such as precalciner kilns and waste heat recovery systems, have shown 

promise in reducing energy consumption and CO2 emissions (Zhang _et al.,_ 2019). Additionally, the use of alternative fuels, such as biomass and waste-derived fuels, can help decrease reliance on fossil fuels and mitigate the environmental impact of clinkerization. 

The clinkerization process plays a central role in cement manufacturing but has significant environmental consequences, particularly in terms of CO2 emissions and energy consumption. Addressing these challenges requires a multifaceted approach, including the adoption of sustainable practices, implementation of energy-efficient technologies, and exploration of alternative fuel sources. By prioritizing sustainability and innovation, the cement industry can mitigate the environmental impact of clinker production and contribute to a more sustainable future. 

## **2.1.3  Quantification of the Environmental Impact of Cement Production** 

The environmental impact of cement production is substantial, primarily due to the energy-intensive nature of the process and the inherent chemical reactions involved. Quantifying the environmental impact of the cement industry, particularly in CO2 emissions and energy use, has been a focal point of research in recent years. Various studies have employed life cycle assessment (LCA) methodologies to assess the carbon footprint and energy consumption associated with cement production. 

For example, the study by Wang _et al._ (2019) conducted an LCA of cement production in China and found that CO2 emissions accounted for the majority of environmental impacts. Similarly, research by Cai _et al_ . (2020) utilized LCA to evaluate the environmental performance of different cement production processes, highlighting the significance of energy consumption in determining environmental impact. The most significant environmental concerns associated with cement production are: 

1. **CO** 2 **Emissions:** The calcination of limestone, a key raw material in cement production, releases large amounts of carbon dioxide (CO2). This process alone accounts for approximately 50% of the total CO2 emissions from cement production. The combustion of fossil fuels to heat the kiln further contributes to CO2 emissions, making the cement industry a major contributor to global greenhouse gas emissions. In fact, according to the European Cement Association (CEMBUREAU), "Co-processing offers a high potential for the cement industry to reduce global CO2 emissions" through the utilization of waste materials as alternative fuels. 

2. **Energy Consumption:** Cement production is an energy-intensive process, requiring significant amounts of thermal and electrical energy. The clinkerization process, in particular, demands high temperatures, typically achieved by burning fossil fuels. The energy consumption in cement production not only contributes to greenhouse gas emissions but also puts a strain on energy resources. The report by CEMBUREAU highlights that in 2006, the European cement industry used an energy equivalent of about 26 Mt of coal for cement production. 

## **2.1.4 Current Sustainability Challenges in the Cement Industry** 

The cement industry faces several sustainability challenges that need to be addressed to mitigate its environmental impact and ensure long-term viability. These challenges include: 

**Reducing CO2 Emissions:** The most pressing challenge is to significantly reduce CO2 emissions from cement production. This requires a multi-pronged approach, including the use of alternative fuels, the development of more energy-efficient technologies, and the exploration of carbon capture and utilization (CCU) technologies. The CEMBUREAU report emphasizes that the use of alternative fuels, such as biomass and waste-derived fuels, can significantly contribute to reducing fossil fuel dependency and lowering emissions. 

**Improving Energy Efficiency:** Enhancing energy efficiency throughout the cement production process is crucial for reducing both CO2 emissions and operational costs. This involves optimizing kiln operations, utilizing waste heat recovery systems, and adopting energy-efficient technologies in all stages of production. 

Addressing these sustainability challenges requires a collaborative effort from all stakeholders, including cement manufacturers, policymakers, researchers, and consumers. By working together, the cement industry can transition towards a more sustainable future, balancing the need for essential construction materials with environmental responsibility. 

## **2.2 Sustainable Practices in Cement Production** 

## **2.2.1 Overview of sustainable practices and innovations in the cement industry:** 

The cement industry is becoming more sustainable by using alternative fuels (like biomass and waste), alternative raw materials (like fly ash and slag), and carbon capture technologies. Digitalization and AI are also improving energy efficiency and reducing waste. 

## **2.2.2 Review of existing measures and technologies aimed at reducing the environmental** 

## **impact of the cement production process** 

Existing measures and technologies aimed at reducing the environmental impact of cement production primarily focus on alternative fuels, raw materials, and carbon capture and utilization (CCU) technologies. 

The cement industry is implementing sustainable practices by utilizing alternative fuels like waste materials and by-products, and exploring carbon capture and storage technologies to minimize its environmental impact and reduce reliance on fossil fuels. 

## **2.2.2 Case studies of successful implementations of sustainable practices.** 

Several case studies highlight successful implementations of sustainable practices in the cement industry: 

- i. National Cement Company Ltd (NCCL) in Kenya: The International Finance Corporation (IFC) supported NCCL in increasing its clinker and cement-grinding capacity while introducing new cement capacity in Uganda. This investment allowed NCCL to utilize the countries' abundant natural resources, boosting local economies and job creation. Additionally, the project focused on lowering carbon emissions in cement production through the use of more efficient technology (IFC, 2020). 

- ii. Cimenterie de Lukala S.A. (CILU) in the Democratic Republic of Congo: IFC's equity investment in CILU, the largest cement manufacturer in the country, facilitated the expansion of cement production capacity. This expansion aimed to create a reliable domestic supply of cement, supporting housing and infrastructure development, generating employment opportunities, and strengthening the local supply chain (IFC, 2020). While the case study doesn't explicitly detail the sustainable practices implemented, the focus on local resource utilization and infrastructure development suggests a commitment to sustainable growth. 

These case studies demonstrate the diverse approaches being taken by cement companies to integrate sustainable practices into their operations. They highlight the potential for collaboration between industry, financial institutions, and governments to drive positive environmental and economic outcomes in the cement sector. 

## **2.3. Process Optimization in Cement Manufacturing** 

## **2.3.1 Importance of process optimization in achieving sustainability:** 

Process optimization plays a pivotal role in the cement industry's pursuit of sustainability. By enhancing efficiency and reducing waste, optimization strategies can significantly lower the environmental impact of cement production while maintaining or even improving product quality and profitability. 

## **Historical approaches to process optimization in the cement industry** 

Historically, process optimization in the cement industry has been a reactive and iterative process, relying heavily on human expertise and experience. Operators and engineers would manually adjust process parameters based on their observations and knowledge of the system. This approach, while effective to some extent, was often time-consuming, labor-intensive, and prone to human error (Arantes, 2021). 

Early automation attempts used simple control loops, like PID controllers, to maintain process variables. Statistical process control (SPC) monitored variations, but didn't proactively optimize the process. 

Model-based control (MBC) strategies, using mathematical models, improved prediction and control. However, developing and maintaining accurate models for the complex cement production process was challenging. 

## **2.4 Machine Learning in Industrial Process Optimization** 

In recent years, machine learning (ML) has emerged as a powerful tool for industrial process optimization. ML algorithms can learn from historical data to identify patterns and 

relationships that are not easily discernible by humans. This knowledge can then be used to optimize process parameters and improve efficiency, quality, and safety. 

One of the key advantages of ML is its ability to handle large and complex datasets. In the cement industry, for example, vast amounts of data are generated from sensors and control systems throughout the production process. ML algorithms can analyze this data to identify inefficiencies and opportunities for improvement. 

For instance, ML has been used to optimize kiln operations in cement plants. By analyzing data from kiln sensors, ML algorithms can identify the optimal operating conditions for maximum energy efficiency and clinker quality. This can lead to significant cost savings and reductions in greenhouse gas emissions. 

ML has also been used to optimize the blending of raw materials in cement production. By analyzing the chemical composition of the raw materials, ML algorithms can determine the optimal blend proportions to achieve the desired cement properties. This can improve the quality and consistency of the cement while reducing the need for expensive additives. 

**2.4.1 Overview of machine learning and its relevance to process optimization:** Machine learning (ML), a subset of AI, enables computers to learn from data and improve their performance without explicit programming. In industrial process optimization, ML enhances efficiency, reduces waste, and improves product quality by analyzing data from various sources and optimizing process parameters in real-time. The relevance of ML to process optimization lies in its ability to handle complex data and identify patterns that may not be apparent to human experts, unlocking new levels of efficiency, productivity, and sustainability. 

Machine learning has high level relevance in the process industry and some of the  areas of successful applications of machine learning in optimizing processes include: 

- i. **Predictive Maintenance** : ML algorithms analyze historical data to predict equipment failures before they occur, allowing for timely maintenance. This minimizes downtime and extends the lifespan of machinery, contributing to cost savings and operational efficiency . 

- ii. **Process Control and Automation** : ML models can automate process control by continuously learning from data and optimizing process parameters in real-time. This reduces human intervention, enhances precision, and improves consistency in production **Energy Efficiency** : ML can identify patterns in energy consumption and recommend adjustments to reduce waste. For instance, it can optimize heating and cooling cycles in cement kilns, leading to significant energy savings and reduced CO₂ emissions. 

## **2.4.2 Advantages of using machine learning for process optimisation** 

Machine learning (ML) can optimize processes by handling large datasets and identifying non-linear relationships. It adapts to changes, enabling real-time optimization and data-driven decisions. 

Specific advantages include: 

- i. Improved energy efficiency and sustainability 

- ii. Predictive capabilities for proactive maintenance and optimization 

- iii. Adaptability and continuous improvement 

- iv. Reduced operational costs 

- v. Scalable solutions that adapt to changes 

Overall, machine learning's data-driven approach and advanced algorithms significantly improve efficiency, sustainability, and profitability in industries like cement production. 

## **2.5 Applications of Machine Learning in Cement Production** 

## **2.5.1 Review of existing studies that have applied machine learning in cement** 

## **production:** 

Machine learning (ML) has increasingly found its place in various industrial applications, with cement production being a notable sector benefiting from its adoption. The application of machine learning in cement production is still an emerging field, but several studies have demonstrated its potential to optimize various aspects of the manufacturing process. These applications can be broadly categorized into: 

1. **Process Control and Optimization:** ML algorithms can be used to model complex relationships between process variables and optimize kiln operation, raw material blending, and grinding processes. For example, a study by Eijkens _et al._ (2016) demonstrated the use of artificial neural networks (ANNs) to model a complete solar cell production line with 49 input variables, predicting cell efficiencies with high accuracy. This approach can be adapted to cement production, where ML models can optimize kiln temperatures, fuel consumption, and raw material ratios to improve efficiency and reduce emissions. 

2. **Predictive Maintenance:** ML algorithms can analyze sensor data from equipment to predict potential failures and schedule maintenance proactively. This can prevent costly breakdowns and improve overall plant reliability. In the cement industry, predictive maintenance can be applied to critical equipment such as kilns, mills, and conveyors, ensuring their optimal performance and longevity. 

3. **Quality Control:** ML models can be trained to predict cement quality based on process parameters and raw material properties. This can help manufacturers maintain consistent product quality and meet customer specifications. For instance, ML algorithms can be used to predict the compressive strength of cement, allowing for adjustments to the production process to ensure the desired strength is achieved. 

4. **Energy Management:** ML can optimize energy consumption in cement plants by analyzing energy usage patterns, identifying inefficiencies, and recommending 

optimal settings for equipment. This can lead to significant cost savings and reduce the environmental impact of cement production. 

Not only this, but even ML can also be applied in the optimization process of all stages of cement manufacturing, starting from the blending process to the very last of it. 

## **Kiln Operation Enhancement** 

The kiln is a critical component in cement production where raw materials undergo complex chemical reactions to form clinker. Maintaining consistent kiln operations is challenging due to the variability in raw materials and the intricate dynamics of the process. Machine learning models have been employed to predict and control these variables more effectively. According to reports, the implementation of MPC in kiln operations has led to significant improvements, such as increased throughput by 2-4%, reduced fuel consumption by 1-3%, and decreased NOx emissions by up to 20% . These optimizations not only enhance operational efficiency but also contribute to environmental sustainability by lowering emissions. [Juliano de Goes Arantes, 2022 AUTOMATION & PROCESS CONTROL] 

## **2.5.2 Predictive Maintenance and Optimization in Cement Plants** 

Predictive maintenance (PdM) is a proactive maintenance strategy that utilizes data analysis and machine learning (ML) to predict equipment failures before they occur. By identifying potential issues early on, PdM enables timely interventions, preventing costly breakdowns and optimizing maintenance schedules (Jardine _et al.,_ 2006). In the cement industry, where equipment reliability is crucial for continuous operation and high productivity, PdM offers significant benefits in terms of cost savings, improved safety, and enhanced sustainability. 

Predictive maintenance is crucial in the cement industry due to the high cost and critical nature of the equipment involved. By implementing predictive maintenance, cement plants can achieve significant cost savings, improve operational efficiency, and ensure continuous 

production without unexpected interruptions. Moreover, it helps in optimizing the use of resources and enhancing safety by preventing catastrophic failures. 

## **2.5.3 Development of Machine Learning Models for Predictive Maintenance** 

The development of ML models for predictive maintenance in a cement plant typically involves the following steps: 

1. **Data Collection:** Sensor data from equipment, such as vibration, temperature, pressure, and acoustic emissions, is collected over time. This data can be supplemented with historical maintenance records, operational logs, and environmental data (Gröning _et al.,_ 2022). 

2. **Data Preprocessing:** The collected data is cleaned, filtered, and transformed to ensure its quality and consistency. This may involve removing outliers, handling missing values, and scaling the data to a suitable range. 

3. **Feature Engineering:** Relevant features are extracted from the preprocessed data. These features may include statistical measures, frequency domain features, or time-domain features that capture the underlying patterns and trends in the data. 

4. **Model Selection and Training:** An appropriate ML algorithm is selected, and the 

model is trained on the prepared dataset. Algorithm choice depends on the specific problem and the characteristics of the data. Common algorithms used for PdM include decision trees, random forests, support vector machines (SVMs), and neural networks (Gröning _et al.,_ 2022). 

5. **Model Evaluation and Validation:** The trained model is evaluated on a separate validation dataset to assess its performance and generalization capabilities. Performance metrics such as accuracy, precision, recall, and F1 score are used to quantify the model's effectiveness. 

## **2.6 Challenges and Limitations of Machine Learning in Cement Production** 

The implementation of machine learning (ML) in cement production, while promising, is not without its challenges and limitations. These can be broadly categorized into technical and practical challenges. 

## **2.6.1 Technical challenges in implementing machine learning** 

The implementation of ML in the cement industry faces practical challenges. These include data quality issues, the complexity and interpretability of models, computational resource requirements, cost and ROI considerations, organizational culture and change management, integration with existing systems, cybersecurity concerns, lack of expertise, and regulatory compliance issues. 

Addressing these challenges and limitations requires a multi-faceted approach. This includes investing in data infrastructure and quality, developing more interpretable ML models, exploring cloud-based solutions to reduce computational costs, and fostering a culture of innovation and collaboration within the cement industry. By overcoming these barriers, the cement industry can fully harness the potential of machine learning to achieve its sustainability goals. 

## **2.7. Research Gap and Current Works:** 

This research tackles the pressing need for sustainable practices within the cement industry, a major contributor to global CO2 emissions and energy consumption. Unlike previous studies that often focus on isolated aspects of kiln optimization, this work develops a comprehensive, data-driven framework that simultaneously addresses the multiple, and often competing, objectives of minimizing energy consumption, reducing CO2 emissions, and maintaining or increasing clinker production rates. The novelty lies in the integrated approach. 

## **CHAPTER THREE** 

## **MATERIALS AND METHOD** 

## **3.1 Materials** 

**3.1.1 Laptop** : The most important material for the research, all the work, simulation and data processing will be carried out with it, including post-data collection simulations and result analysis. 

**3.1.2 Plant Dataset:** In order to make a clear understanding of the plant process variables 

dataset was collected from the Ashaka Cement Plant, by Lafarge Holcim in other to explore the relationship between the process data and understand their dependencies on each other and how they collectively affect the plant and their individual contribution to the plant’s dynamics. 

The Plant has two rotary kilns in operation, two of which have a  max capacity of 50tph output, they both depend on their individual raw mill and ID fan (induced draft fan) and work mostly at non-optimal state, the data gotten from the plant for power consumption include, Power (Kw), Power Consumption (MWH), Power Consumed (KWt/h), Absorbed Power (Kw), Power Consumption (Kw), Max Power (rotary drive) Kw, Energy Consumption (kWh/t-clinker), aside this, the dataset include process variables such as, Kiln Temperature (°C), Fuel Consumption (kg/h), Raw Meal Composition (% Limestone), Clinker Production Rate (tons/h), Oxygen Levels (%), Kiln Pressure (bar), Air Flow Rate (m³/h), Fuel, Emissions (kg/h), the kiln process flow diagram is shown in figure 3.1 below. 



<!-- Start of picture text -->
PADRE ep omewareR KILN OPERATION @ rerurenunen uae<br>Seren owes 4<br>=— — ' _—<br>=N Z| H !v 3<br>Ete sere i WARE Ay 5<br>gO cattin i<br>ae a Biel eS al<br>ots a eeee 6 |<br>SAobo leSl os Pa<br>U V7 OU ai] ko<br>rn = W te Te<br><!-- End of picture text -->

**Figure 3.1:** Image showing the rotary kiln flow diagram 

The dataset obtained from the plant was also scaled to have a million line data, 80% of which 

is set aside for training and the remaining 20% for validation and testing datasets. 

The tables below show the first and last five lines of the datasets. 

## **3.2. Method** 

This project focuses on modifying machine learning algorithms to create a model that accurately predicts kiln energy consumption and CO2 emissions. The results are displayed in the form of datasets and charts. 

## **3.2.1 Application of M L Models** 

Random Forest Regressor (RFR) is a supervised learning algorithm based on an ensemble of regression decision trees (RDT) (Zojaji, _et al_ . 2022). Introduced by Breiman (Breiman 2001), RFR extends the Bootstrap Aggregating (Bagging) framework (Hou, _et al_ . 2022) to regression tasks. The algorithm constructs multiple RDTs trained on bootstrapped subsets of the training data, sampled with replacement, and aggregates their predictions to produce a final output. During tree construction, RFR randomly selects a subset of input features at each node split, decorrelating individual trees and enhancing generalization (Chelgani 2019). 

For regression, each RDT partitions the feature space to minimize the Mean Squared Error (MSE) at internal nodes. The MSE for a node (t) is computed as: 



(3.1) 

where ( Nt ) is the number of training samples in node ( t ), ( yi ) is the target value of the ( i )-th sample, and: 

𝑡 = 𝑁𝑡 1 ∑𝑖∈𝑡 𝑦𝑖 

## (3.2) 

is the mean target value within ( t ). Splits are chosen to maximize the reduction in 

MSE: 





where tleft  and tright are the child nodes. 

The final prediction of the RFR is the arithmetic mean of predictions from all ( N ) RDTs: 



(3.4) 

where ( x ) is the input feature vector, and 𝑇𝑛 (𝑥) denotes the prediction of the n<sup>-th</sup> regression 

tree (Wager _et al_ 2018; J. Zhou _et al_ . 2019). 

Key advantages of RFR include: 

1. Reduced overfitting through variance reduction (Gong _et al_ . 2018), 

2. Robustness to outliers and noisy data (Chelgani, Nasiri, and Tohry 2021), 

3. Low bias due to hierarchical feature space partitioning (Nasiri, Homafar, and Chelgani 2021), 

4. Minimal hyperparameter tuning requirements (Ouallouche, Lazri, and Ameur 2018). 

## **3.2.2 Multi Objectives Optimization Models Application** 

NSGA-II is a popular multi-objective evolutionary algorithm used to solve multi-objective optimization problems. Proposed by Deb _et al_ . (2002), NSGA-II improves upon the original NSGA by addressing its high computational complexity, lack of elitism, and need for a 

sharing parameter. NSGA-II uses a fast non-dominated sorting approach, crowding distance estimation, and a simple crowded comparison operator to achieve better convergence and diversity in solutions. The algorithm evolves a population of candidate solutions towards the Pareto-optimal front. 

The core steps of NSGA-II are as follows: 

i. Initialization: A parent population, Pt , of size _N_ is created randomly. 

ii. Evaluation: Each individual in Pt is evaluated according to the _M_ objective functions. 

iii. Non-dominated Sorting: Pt is sorted into different non-domination levels (fronts), Fi , based on Pareto dominance. An individual _p_ dominates another individual _q_ (denoted as p≺q) 

if and only if: 









iv. Where fi (x) represents the value of the _i_ -th objective function for solution _x_ . The first front, F1 , contains all non-dominated individuals. Subsequent fronts are created by removing the individuals in the previous fronts and repeating the non-domination sorting on the remaining population. Formally: 





_and subsequent fronts, are constructed by removing the individuals of previously found._ 





And, iterating Non-dominated sorting untill, 



iv. Crowding Distance Assignment: For each front Fi , a crowding distance is assigned to each individual _p_ . The crowding distance, d(p), measures the density of solutions surrounding _p_ . It is calculated as the average distance of two points on either side of this point along each of the objectives. For the _m_ -th objective, let _I_ be the sorted set of individuals in Fi according to the _m_ -th objective, and let I[j] be the _j_ -th individual in this sorted set. The crowding distance contribution from the _m_ -th objective is: 





𝑚𝑎𝑥 𝑚𝑖𝑛 Where 𝑓𝑚 and 𝑓𝑚 are the maximum and minimum values of the _m_ -th objective function 

in the current population (or, ideally, in the entire search space, but practically often approximated within the current population). Individuals at the extremes (highest and lowest objective function values) are assigned an infinite crowding distance. The total crowding distance is the sum of the individual objective crowding distances: 





Selection: A binary tournament selection with a crowded comparison operator is used to 

select individuals for the mating pool. The crowded comparison operator (≺n ) is defineas follows: 

- C. If _p_ and _q_ are in different fronts (ip <iq ), then p≺n q (where ip is the front number of _p_ ). 

- D. If _p_ and _q_ are in the same front (ip =iq ), then p≺n q if d(p)>d(q). 

- iii. This operator favors individuals in lower (better) non-domination ranks and, among individuals in the same rank, favors those in less crowded regions. 

vi. Crossover and Mutation: Standard genetic operators (crossover and mutation) are applied to the selected individuals to create an offspring population, Qt , of size _N_ . 

vii. Combination: The parent population (Pt ) and the offspring population (Qt ) are combined to form a combined population of size 2 _N._ 

𝑅𝑡 = 𝑃𝑡 ∪𝑄𝑡 (3.11) 

viii. Elitism and Truncation: Rt is sorted based on non-domination, and the crowding distance is calculated for each front. The new parent population, Pt+1 , is formed by selecting the best _N_ individuals from Rt based on the crowded comparison operator. This ensures elitism by preserving the best solutions from the previous generation. The fronts are added to Pt+1 until its size equals or exceeds _N_ . If a front cannot be completely added without exceeding size _N_ , individuals from that front are selected based on their crowding distance, favoring those with larger distances. 

ix. Iteration: Steps 2-8 are repeated until a termination criterion (e.g., a maximum number of generations) is met. 

The final output of **NSGA-II** is the population Pt at the termination, which approximates the Pareto-optimal set. 

The main advantage of the NSGA-II are that it’s Fast non-dominated sorting approach, elitism and diversity preservation using a crowding distance. (Zhang, Zhou, and Zhang 2008; Li _et al_ . 2015) 

# **3.2.3 Model simulation and validation** 

To rigorously evaluate the performance of the prediction models and the effectiveness of the optimization, a comparative analysis was conducted, summarized in Table 4.3. This table juxtaposes the model's predicted values (based on average input features), the optimized values (from a balanced Pareto-optimal solution), and the actual plant values (represented by the median of the historical data). Furthermore, for each parameter, the percentage deviation and Root Mean Squared Error (RMSE) are calculated, using the validation of the models with plant and literature data (Ojong _et al_ ., 2024; Wordu _et al_ ., 2022) 



where, N is the number of iterations of model results 



comparing both the model predictions and the optimized values to the corresponding plant values. 

## **3.3 Input Parameters** 

This section details the input parameters used in the machine learning models and the subsequent multi-objective optimization. These parameters represent the controllable and measurable variables that influence the cement kiln's operation and performance. The parameters are categorized for clarity, and their corresponding units are provided. 

|Category|Parameter|Value Range|Unit|
|---|---|---|---|
|Process<br>Variables|Kiln Temperature|1342.05 - 1368.85|°C|



||Fuel Consumption|184.34 - 223.06|kg/h|
|---|---|---|---|
||Raw Meal Composition (%<br>Limestone)|80.24 - 86.80|%|
||Clinker Production Rate|80.24 - 99.58|tons/h|
||Oxygen Levels|3.39 - 4.12|%|
||Air Flow Rate|1750.94 - 2219.44|m³/h|
||Kiln Pressure|13.71 - 16.20|bar|
|Power<br>Related|Power (Kw)|4010.71 - 5953.36|Kw|
||Power Consumption (MWH)|325128.60<br>-<br>443202.02|MWH|
||Power Consumed (KWt/h)|4062.84 - 5618.01|KWt/h|
||Absorbed Power (Kw)|3980.41 - 5079.33|Kw|
||Power Consumption (Kw) Max|6119.53 - 7815.66|Kw|
||Power (rotary drive) Kw|520.54 - 764.82|Kw|
|Engineered<br>Features|Theoretical CO2|Calculated|kg/h|
||Theoretical Energy|Calculated|Kw|
||Heat Efficiency|Calculated|-<br>(dimensionl<br>ess)|
||O2 Deviation|Calculated|%|



## **3.4 Solution Techniques** 

The algorithm used in solving ML model with Random Forest Regressor and solving the multi-objective optimisation with NSGA-II are stated as thus: 

## **Step 1: Data Preprocessing with Chemical Engineering Features** 

The preprocessing step involves adding domain-specific features to the dataset using chemical engineering principles. These features are derived from the raw data and are critical for modeling the cement production process. 

## **i. Theoretical CO2 Calculation:** 

The theoretical CO2 produced from limestone calcination (CaCO3 →CaO+CO2 ) is calculated as: 

Tco2 = RMC (% caco3)   CPR  0.44 × × 

(3.13) 

Where RMC is Raw mill composition in percentage of limestone, and CPR is Clincker production rate in tons/hour and Tco Here, 0.44 is the stoichiometric factor for CO2  production from limestone. 

## **ii. Heat Balance Calculations:** 

The theoretical energy required for calcination is computed using the enthalpy of calcination: 

𝐶𝑃𝑅 × 1000 × 𝐶𝐻 TE    = 3600 

(3.14) 

TE    = 0. 278(𝐶𝑃𝑅× 𝐶𝐻) 

(3.15) 

Where TE  is Theoretical energy of the system, CPR is the Clincker production rate and CH is the Calcination enthalpy. 

The heat efficiency is then calculated as: 

𝐻𝐸 =<sup>𝑇𝐸</sup> 𝐹𝐶 

(3.16) 

Where HE represents the Heat efficiency, TE is the Theoretical energy and FC is the Fuel consumption measured in Kg/h. 

## **iii. Combustion Efficiency Indicator:** 

The deviation from the optimal oxygen level (4%) is calculated as: 

O2 Deviation = % O2  Level - 4.0 

(3.17) 

## **Step 2: Machine Learning Model Training** 

A Random Forest Regressor is trained to predict energy consumption and CO2 emissions. 

The training process involves the following steps: 

## **i. Feature Selection:** 

The features used for training include process variables (e.g., kiln temperature, fuel consumption) and the engineered features (e.g., theoretical CO2 , heat efficiency). 

## **ii. Data Splitting:** 

The dataset is split into training and testing sets using an 80-20 split: 

𝑋𝑡𝑟𝑎𝑖𝑛 , 𝑋𝑡𝑒𝑠𝑡 , 𝑦𝑡𝑟𝑎𝑖𝑛 , 𝑦𝑡𝑒𝑠𝑡 = 𝑡𝑟𝑎𝑖𝑛_𝑡𝑒𝑠𝑡_𝑠𝑝𝑙𝑖𝑡(𝑋, 𝑦, 𝑡𝑒𝑠𝑡_𝑠𝑖𝑧𝑒= 0. 2, 𝑟𝑎𝑛𝑑𝑜𝑚_𝑠𝑡𝑎𝑡𝑒= 42) 

(3.18) 

## **iii. Model Training:** 

The Random Forest Regressor is trained on the training data: 

𝑚𝑜𝑑𝑒𝑙. 𝑓𝑖𝑡(𝑋𝑡𝑟𝑎𝑖𝑛 , 𝑦𝑡𝑟𝑎𝑖𝑛 ) 

(3.19) 

## **iv. Model Evaluation:** 

The model's performance is evaluated using the coefficient of determination (R2): 

𝑅2 = 𝑚𝑜𝑑𝑒𝑙. 𝑠𝑐𝑜𝑟𝑒(𝑋𝑡𝑒𝑠𝑡 , 𝑦𝑡𝑒𝑠𝑡 ) (3.20) 

## **Step 3:  Multi-Objective Optimization (NSGA-II)** 

The optimization problem is formulated to minimize both energy consumption and CO2 

emissions while maintaining production constraints. 

## **i. Problem Definition** : 

The optimization problem is defined as: 

Min F(X) = [ EC, CO2 E] (3.21) 

Where EC is Energy Consumption and CO2 E is CO2 emission is subjected to: 

g(x) = CuPR − 𝑥3 ≤ 0 (3.22) 

Here CuPR represents Current production rate and x3 represents the clinker production rate, 

and x is the vector of decision variables. 

## **ii. NSGA-II Algorithm:** 

The NSGA-II algorithm is used to solve the multi-objective optimization problem. It works as follows: 

- a. Initialize a population of candidate solutions. 

- b. Evaluate the objectives and constraints for each candidate. 

- c. Perform non-dominated sorting and crowding distance calculation to select the best 

   - solutions. 

- d. Use crossover and mutation to generate new solutions. 

- e. Repeat for a specified number of generations. 

iii.  Optimization Execution: 

The optimization is executed using the minimize function: 

𝑟𝑒𝑠𝑢𝑙𝑡= 𝑚𝑖𝑛(𝑝𝑟𝑜𝑏𝑙𝑒𝑚, 𝑎𝑙𝑔𝑜𝑟𝑖𝑡ℎ𝑚, (𝑛𝑔𝑒𝑛, 100), 𝑠𝑒𝑒𝑑= 42, 𝑣𝑒𝑟𝑏𝑜𝑠𝑒= 𝐹𝑎𝑙𝑠𝑒)    (3.23) 

## **Step 4:  Recommendation Engine with Process Constraints** 

The recommendation engine uses the optimization results to suggest optimal process parameters while adhering to operational constraints. 

## i. Constraint Handling: 

The engine ensures that the recommended parameters do not violate the production rate constraint: 

𝑔(𝑥) = 𝐶𝑢𝑃𝑅−𝑥3 ≤ 0 

(3.24) 

ii. Optimal Parameter Selection: 

The Pareto-optimal solutions from the NSGA-II result are analyzed to select the best trade-off between energy consumption and CO2 emissions. 



<!-- Start of picture text -->
DATA LOADING AND PREPROCESSING<br>dd engineering. fotos:7 4 engineering features: Merge energy_data<br>Energy Data ~ and co2_data<br>MODEL TRAINING<br>=<br>energy_model.train c02_model.train<br>= =<br>MULTIOBJECTIVE OPTIMIZATION<br>A ray<br>Conergy mode eomode) Initializealgorithm NSGA2<br>Pareto-Optimal Solutions<br>VISUALIZATION AND REPORTING<br><!-- End of picture text -->

**Figure 3.2: Code Structure and Algorithm** 

**CHAPTER FOUR** 

## **RESULTS** 

## **4.1 Results** 

The goal of this research is to find optimal operating conditions for a cement kiln to minimize both energy consumption and CO2 emissions. The expected outcome is a set of Pareto-optimal solutions that represent different trade-offs between these two competing objectives. This will enable data-driven decision-making, quantifiable trade-offs, operational flexibility, improved process understanding, benchmarking, target setting, and a contribution to sustainability in cement production. 

## **4.1.1 Results from Prediction Models** 

The accuracy of the Random Forest Regressor (RFR) models, trained to predict energy consumption and CO2 emissions using preprocessed data from the cement kiln, is crucial for the multi-objective optimization. The R² value was used to assess the model performance. 

The results of the model training are as follows: 

The energy consumption and CO2 emission predicted values are 1162 KWh/t-clinkeer and 10005.78 kg/hr of CO2 emission when the clicker production rate is around 91.80 tph, these are accurately predicted before surge happens in the process, and the value predicted are reliable based on the R<sup>2</sup> value achieved for both energy consumption random forest model and that of CO2 emission of value 0.98 for both of them. Figure 4.1 shows the R<sup>2</sup> of determination chart for the data results. 



**Figure 4.1:** R² value chart for both the energy consumption and CO2 emission chart. 

CO2 emission and rates shows accuracy and decency in dataset using RFR algorithm, Figure 

4.2 and 4.3 shows the feature importance chart amongst all parameters considered for the 

work, showing the hierarchy of importance of all values. 



<!-- Start of picture text -->
Feature Importance: Energy Consumption Model<br>0s<br>04<br>8<br>&<br>£03<br>5<br>8<br>—<br>02<br>ol<br>0.0 ee 2 ts %* £€GS FE RPE ETN Bz eE<br>=e. ec. et & © eo ee EE EAE 2<br>Se ee ¢ es F 58% FS 5 FBS =<br><!-- End of picture text -->

**Figure 4.2:** Feature importance for the energy consumption prediction model. The height of each bar represents the relative importance of the corresponding feature. 



<!-- Start of picture text -->
Feature Importance: CO2 Emissions Model<br>0.40<br>0.35<br>0.30<br>0.25<br>e<br>&<br>£ 0.20<br>5<br>8<br>—<br>0.15<br>0.10<br>0.05<br>0.00<br>2e2enae0233353285<br>F5886 fF F227 F ssrBFERREEEEBE<br>& &€ = 3 2» € gee 28 > 8588<br>4 ¢ 8 5 «© & & 8 g$ 2 £€ 5 ¢ Fs Fz<br><!-- End of picture text -->

**Figure 4.3:** Feature importance for the CO2 emissions prediction model. The bar heights represent the relative importance of each feature. 

The most important features for predicting energy consumption are 'Power Consumed (KWt/h)' and 'Absorbed Power (Kw)'. Other power-related features also show importance. Controlling key power parameters is critical for minimizing energy consumption, and the indirect influence of other process variables on power draw should be investigated. 

The most influential features on CO2 emissions are Clinker Production Rate and Theoretical Energy. Fuel Consumption and Theoretical CO2 are secondary factors, while Kiln Temperature, Air Flow Rate, Heat Efficiency, and Power have minimal direct impact. 

## **4.2 Optimization of the parameters** 

The core result of the multi-objective optimization is the Pareto front, which visually represents the trade-off between energy consumption and CO2 emissions. Figure 4.1 displays the Pareto front obtained from the NSGA-II algorithm. 



<!-- Start of picture text -->
Pareto Front: Energy Consumption vs CO2 Emissions<br>ec. @ Pareto Front<br>920<br>918<br>€<br>g<br>e<br>2°<br>S<br>8= 916 * e<br>&e<br>8<br>6 ee<br>gia e e<br>%e<br>%e<br>912 ees<br>1030 1035 1040 1045 1050<br>Energy Consumption (kwh/t-clinker)<br><!-- End of picture text -->

**Figure 4.4:** Effect of energy consumption rate on CO2 emission rate 

The Pareto front exhibits a clear, generally decreasing trend, indicating that reducing CO2 emissions typically requires an increase in energy consumption, and vice-versa. This is a fundamental characteristic of many multi-objective optimization problems, where improving one objective often comes at the cost of another. However, the _shape_ of the Pareto front and the _distribution_ of points along it provide crucial insights. 

## **4.2.1 Pareto front Interpretation:** 

Non-Linear Trade-off **:** The trade-off is not perfectly linear. The curve has regions of varying steepness. For instance, moving from approximately 1052 kWh/t-clinker to 1045 kWh/t-clinker on the x-axis results in a relatively small increase in CO2 emissions (from about 912 kg/h to 914 kg/h). However, moving from 1035 kWh/t-clinker to 1030 kWh/t-clinker results in a much larger increase in CO2 emissions (from about 916 kg/h to over 920 kg/h). This non-linearity is important for decision-making. It suggests that in certain operating ranges, significant reductions in energy consumption can be achieved with minimal impact on CO2 emissions, while in other ranges, the trade-off is much more severe. 

The Pareto front shows clusters of solutions, especially at the lower-energy/higher-CO2 and higher-energy/lower-CO2 ends. The range of optimal solutions for energy consumption is 1028 kWh/t-clinker to 1053 kWh/t-clinker, and for CO2 emissions is 912 kg/h to 921 kg/h. The choice of operating point depends on priorities: cost minimization favors lower energy consumption, emissions reduction favors lower CO2 emissions, and a balanced approach finds a compromise between the two. 

## **4.2.2 Optimal Values** 

Table 4.1 provides a representative subset of three Pareto-optimal solutions extracted from the optimization results. These solutions were chosen to illustrate different points along the Pareto front: 

**i. Solution 1:** Represents a point towards the lower energy consumption end of the spectrum (higher CO2 emissions). 

**ii. Solution 25:** Represents a point roughly in the middle of the Pareto front, offering a balance between the two objectives. 

**iii. Solution 49:** Represents a point towards the lower CO2 emissions end of the spectrum (higher energy consumption). 

**Energy Consumption vs. CO2 Emissions:** As expected, there's a trade-off. Solution 1 (low energy) has the lowest energy consumption (1029.18 kWh/t-clinker) but the highest CO2 emissions (920.55 kg/h). Solution 49 (low CO2) has the lowest CO2 emissions (913.51 kg/h) but the highest energy consumption (1049.82 kWh/t-clinker). Solution 25 is a compromise. 

**Clinker Production Rate:** Solution 49, focused on minimizing CO2, has the _highest_ clinker production rate (89.95 tons/h). This is a key insight. It suggests that, within the optimized parameter space, _higher_ production rates can, under certain conditions, lead to _lower_ CO2 emissions _per unit of production_ . This is likely due to improved thermal efficiency at higher throughputs (spreading the fixed energy costs over more product). This effect outweighs the increased _total_ CO2 from processing more raw material. 

**Fuel Consumption:** Solution 49 (low CO2) has a _higher_ fuel consumption (228.87 kg/h) than the other solutions. This seems counterintuitive, but it's coupled with the higher production rate. The energy consumption per ton of clinker is still higher for Solution 49, but the CO2 emissions per hour are lower. This implies a more efficient use of the fuel's energy for clinker production, relative to CO2 produced. 

**Raw Meal Composition (% Limestone):** Solution 49 has the highest limestone percentage (85.42%). While limestone calcination releases CO2, the optimization algorithm has found a combination of parameters where this higher limestone content, combined with the higher production rate and other factors, results in lower overall CO2 emissions per hour. This suggests that the increased efficiency from higher production outweighs the increased CO2 from the slightly higher limestone content. 

**Theoretical CO2 and Energy** : The values for the calculated features reflects the values of Clinker production rate, and Raw meal composition. 

**Oxygen Levels:** Oxygen levels are relatively consistent across the solutions, close to the target of 4%. The optimization algorithm has likely found that small variations in oxygen levels within this range have a less significant impact on the objectives compared to other factors. 

**Kiln Temperature:** Solution 49 has a slightly lower kiln temperature. 

**Power Related parameters** : Solution 25 has the lowest values related to power, while Solution 49 has the lowest Absorbed Power (Kw). 

**Kiln Pressure and Air Flow Rate:** These parameters show some variation, but their impact is likely indirect, influencing the combustion process and heat transfer within the kiln. 

The optimization process has identified complex, non-intuitive relationships between the input features. Simply reducing fuel consumption, for instance, doesn't necessarily lead to the lowest CO2 emissions. The algorithm has found operating points where higher production rates, coupled with specific adjustments to other parameters (fuel consumption, raw meal composition, etc.), can achieve _both_ lower energy consumption _per ton of clinker_ and lower _overall_ CO2 emissions. This highlights the power of multi-objective optimization to uncover non-obvious solutions. 

The table provides concrete operating recommendations. A plant could choose to implement Solution 1, 25, or 49 (or a point in between), depending on its priorities. The results suggest that focusing solely on minimizing fuel consumption might not be the optimal strategy for minimizing CO2 emissions. Increasing the clinker production rate, _in conjunction with_ the other optimized parameter settings, can lead to a more sustainable outcome. This is a critical finding. 

**Table 4.1: Representative Pareto-Optimal Solutions** 

|Feature|Solution 1 (Low<br>Energy)|Solution 25<br>(Balanced)|Solution 49 (Low<br>CO2)|
|---|---|---|---|
|Power (Kw)|4075.44|4056.12|4076.69|
|Power Consumption (MWH)|337909.30|336703.10|434612.12|
|Power Consumed (KWt/h)|4065.20|4063.63|4067.19|
|Absorbed Power (Kw)|4735.52|4716.73|4279.74|
|Power Consumption (Kw)<br>Max|6778.29|6789.87|6119.53|
|Power (rotary drive) Kw|764.82|520.54|639.94|
|Kiln Temperature (°C)|1364.45|1365.58|1353.31|
|Fuel Consumption (kg/h)|194.36|194.36|228.87|
|Raw Meal Composition (%|84.17|84.13|85.42|
|Limestone)||||
|Clinker<br>Production<br>Rate|83.85|83.56|89.95|
|(tons/h)||||
|Oxygen Levels (%)|3.86|3.86|4.13|
|Kiln Pressure (bar)|14.09|13.96|14.14|
|Air Flow Rate (m³/h)|1837.63|1831.20|1878.88|
|Theoretical_CO2|30.73|30.61|33.61|
|Theoretical_Energy|41.14|40.99|44.19|
|Heat_Efficiency|211.69|210.89|193.07|
|O2_Deviation|0.14|0.14|0.14|
|Energy<br>Consumption|1029.18|1030.42|1049.82|
|(kWh/t-clinker)||||
|CO2 Emissions (kg/h)|920.55|920.42|913.51|



## **4.3 Validation of Models** 

Table 4.2 presents the validations of predicted and optimised values from the models against the raw data gotten from the plant, subset of three important parameters to this work, and their Pareto-optimal solutions extracted from the optimization results, model predicted values and the values gotten from the plant. 

- I. **Model Accuracy:** The "Model Predicted Value" column provides a crucial check on the model's accuracy. The deviations from the "Plant Value" are small for all three parameters: 

Energy Consumption (0.63%, 7.34 kWh/t-clinker), CO2 Emissions (3.32%, 32.33 kg/h), and Clinker Production Rate (0.00%, 0.00 tons/h). 

- II. These small deviations and low RMSE values confirm that the trained Random Forest models accurately represent the typical operating conditions of the plant, providing further evidence of their validity (in addition to the R² values reported earlier). 

- III. **Optimization Effectiveness:** The "Optimized Value" column, representing a balanced solution from the Pareto front, demonstrates the potential for improvement 

The optimization led to a significant reduction in energy consumption (11.24%) and CO2 emissions (5.45%), while slightly improving the clinker production rate (0.28%). 

**Table 4.2: Model Validation and Deviation Analysis** 

|**Parameter**|**Model Predicted**<br>**Value**|**Optimized**<br>**Value**|**Plant Value**|
|---|---|---|---|
|Energy Consumption|1168.21|1030.42|1160.87|
|(kWh/t-clinker)||||
|_Deviation (%)_|_0.63_|_-11.24_|-|
|_RMSE_|_0.0734_|_0.0130_|-|
|CO2 Emissions (kg/h)|1005.78|920.42|973.45|
|_Deviation (%)_|_3.32_|_-5.45_|-|
|_RMSE_|_0.0323_|_0.0530_|-|
|Clinker Production Rate|91.80|92.05|91.80|
|(tons/h)||||
|_Deviation (%)_|_-_|_0.28_|-|
|_RMSE_|_-_|_0.25_|-|



## **CHAPTER FIVE** 

## **CONCLUSION AND RECOMMENDATIONS** 

## **5.1 Conclusion** 

The following findings where obtained, 

(i). 0.98 R<sup>2</sup> achieved energy consumption energy consumption rate average of 1168.25 KWh/t-clinker and CO2 emission average of 1005.78 kg/h were obtained from the prediction model. 

(ii).  This optimized gave reduction in energy consumption to 1030.42 kWh/t-clinker (an 11.24% decrease compared to the plant value) and a reduction in CO2 emissions to 920.42 kg/h (a 5.45% decrease), while simultaneously achieving a slight increase in clinker production rate to 92.05 tons/h (a 0.28% increase compared to the plant's median of 91.80 tons/h). 

(iii). The negligible deviation and RMSE values of 0.63%, 3.32%, and 0.00%, and 0.0734, 0.0323, and 0.00 for the Model Predicted Values; and 11.24%, 5.45%, and 0.28%, and 130.44, 0.0530, and 0.0025 for the Optimized Values, respectively, for energy consumption, CO2 emission, and clinker production, confirm the accuracy of the models and the effectiveness of the optimization in achieving significant improvements 

## **5.2 Recommendations** 

i. Incorporate Clinker Quality: Extend the multi-objective optimization to include clinker quality metrics as additional objectives. This would provide a more comprehensive assessment of kiln performance and allow for optimization of quality alongside energy and emissions. This would likely require collecting data on clinker quality parameters (e.g., free lime content, C3S content). 

ii. Dynamic Modeling: Develop dynamic models of the kiln process, rather than the steady-state models used in this research. Dynamic models can capture the transient behavior of the kiln and allow for optimization of control strategies over time. This would be particularly valuable for handling disturbances and transitions between operating points. 

iii. Fuel Type and Composition: Incorporate fuel type and composition as input variables. Different fuels have different carbon intensities, which significantly impacts CO2 emissions. This would allow the optimization framework to recommend not only optimal operating parameters but also optimal fuel choices. 

iv. Uncertainty Quantification: Implement methods for quantifying the uncertainty in the model predictions and the Pareto front. This could involve using techniques like Bayesian modelling or ensemble methods to provide confidence intervals around the optimal solutions. By implementing these recommendations, the cement industry can move towards more sustainable and efficient operations, reducing both energy costs and environmental impact. The combination of machine learning, multi-objective optimization, and real-world validation provides a powerful approach to achieving these goals. 

**REFERENCES** 

- Andrew, R. M. (2018). Global CO₂ emissions from cement production, 1928–2018. Earth _System Science Data_ , 10(2), 195-217. 

- Barker, D. J., Turner, S. A., Napier-Moore, P. A., Clark, M., & Davison, J. E. (2021). Carbon capture and storage for the cement industry. _Greenhouse Gas Control Technologies_ , 10, 387-394. 

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. _Journal of Machine Learning Research,_ 13, 281-305. 

- Breiman, L. (2001). Random forests. _Machine Learning,_ 45(1), 5-32. 

- CEMBUREAU. (2006). The role of cement in the 2050 low carbon economy. _European Cement Association Report._ 

- Chen, X., Li, W., Xu, Y., & Zhang, J. (2019). Application of machine learning for process optimization in cement production. _Journal of Cleaner Production_ , 240, 118200. 

- Chesbrough, H. (2003). Open innovation: The new imperative for creating and profiting from technology. Harvard Business School Press. 

- Coello, C. A. C. (2006). Evolutionary multi-objective optimization: A historical view of the field. _Computational Intelligence Magazine_ , 1(1), 28-36. 

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multi-objective genetic algorithm: NSGA-II. _IEEE Transactions on Evolutionary Computation,_ 6(2), 182-197. 

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. _IEEE Transactions on Evolutionary Computation,_ 6(2), 182-197. 

- Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608. 

- El-Halwagi, M. M. (2012). Sustainable design through process integration: Fundamentals and applications to industrial pollution prevention, resource conservation, and profitability enhancement. Elsevier. 

- Elkington, J. (1997). Cannibals with forks: The triple bottom line of 21st-century business. Capstone. 

- Freeman, C. (1984). The economics of industrial innovation. MIT Press. 

- Habert, G. (2016). Assessing the environmental impact of conventional and ‘green’ cement production. _Advances in Cement Research_ , 28(11), 762-776. 

- Habert, G., & Miller, S. A. (2012). Carbon capture and utilization in the cement industry _. Journal of Cleaner Production,_ 32, 87-94. 

- IFC (International Finance Corporation). (2020). Sustainable cement production: Case studies from developing countries. World Bank Group Report. 

- International Energy Agency (IEA). (2020). Cement technology roadmap 2020. International Energy Agency Report. 

- Li, J., & Zhou, Z. (2020). A novel hybrid model for energy optimization in cement Manufacturing using machine learning techniques. Energy, 192, 116507. 

- Li, M., Yao, X., Chen, Z., Cui, Z., & Zhou, J. (2015). Multi-objective evolutionary algorithms: A survey of the state of the art. _Swarm and Evolutionary_ Computation, 26, 1-14. 

- Lipton, Z. C. (2018). The mythos of model interpretability. Queue, 16(3), 31-57. 

- Najar, M., Kim, H., & Park, J. (2019). Optimization of kiln operation using machine learning. _Journal of Process Control_ , 83, 56-67. 

- Porter, M. E., & van der Linde, C. (1995). Toward a new conception of the environment-competitiveness relationship. _Journal of Economic Perspectives_ , 9(4), 97-118. 

- Powers, D. M. W. (2020). Evaluation: From precision, recall, and F-measure to ROC, informedness, and correlation. _Journal of Machine Learning Technologies_ , 2(1), 37-63. 

- R., Sahu, S., & Patel, R. (2018). Energy efficiency in cement production: A case study approach. _Energy Reports_ , 4, 445-454. 

- Schneider, M., Romer, M., Tschudin, M., & Bolio, H. (2011). Sustainable cement production: Present and future. _Cement and Concrete Research,_ 41(7), 642-650. 

- Scrivener, K., Snellings, R., & Lothenbach, B. (2018). A practical guide to microstructural analysis of cementitious materials. CRC Press. 

- Smith, A., Brown, T., & Miller, D. (2018). Optimization in the cement industry: Current trends and future directions. _Industrial Engineering & Management_ , 7(2), 1-15. 

- Srinivas, N., & Deb, K. (1994). Multiobjective optimization using nondominated sorting in genetic algorithms. _Evolutionary Computation,_ 2(3), 221-248. 

- Towler, G., & Sinnott, R. (2012). Chemical engineering design: Principles, practice, and economics of plant and process design. Elsevier. 

- TTGV (Technology Development Foundation of Turkey). (2023). Innovations in carbon capture for cement production. TTGV Research Report. 

- Wang, H., Zhang, X., & Li, Q. (2021). Real-time process optimization in cement manufacturing using artificial intelligence. _Computers & Chemical Engineering,_ 146, 107239. 

- Zhang, Q., Zhou, A., & Zhang, Y. (2008). Multiobjective optimization: A brief overview. In _Computational Intelligence_ (pp. 395-413). Springer. 

- Zhang, Y., Liu, J., & Wang, B. (2020). Smart kiln operation: Machine learning applications in cement production. _Journal of Process Engineering_ , 92, 45-63. 

- Zhu, X., He, J., & Zhao, L. (2021). Integrating sustainable practices in cement production: A review. _Sustainable Manufacturing & Engineering_ , 23, 112-126. 

