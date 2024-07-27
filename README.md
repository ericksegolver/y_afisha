# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import random
import io


# In[2]:


visits = pd.read_csv('/datasets/visits_log_us.csv')
orders = pd.read_csv('/datasets/orders_log_us.csv')
costs = pd.read_csv('/datasets/costs_us.csv')


# # Verificación de datos

# ## Verificación del dataframe 'visits'

# In[3]:


visits.info()


# In[4]:


visits.columns = visits.columns.str.lower()
visits.columns = visits.columns.str.replace(' ', '_')

visits


# In[5]:


duplicated_rows = visits[visits.duplicated()]

print("Filas duplicadas:")
print(duplicated_rows.count())


# In[6]:


visits['device'] = visits['device'].astype('category')
visits['end_ts'] =  pd.to_datetime(visits['end_ts'], format="%Y-%m-%d %H:%M:%S")
visits['start_ts'] =  pd.to_datetime(visits['start_ts'], format="%Y-%m-%d %H:%M:%S")

visits.info()


# ## Verificación del dataframe 'orders'

# In[7]:


orders.info()


# In[8]:


orders


# In[9]:


orders.columns = orders.columns.str.lower()
orders.columns = orders.columns.str.replace(' ', '_')

orders


# In[10]:


duplicated_rows = orders[orders.duplicated()]

print("Filas duplicadas:")
print(duplicated_rows.count())


# In[11]:


orders['buy_ts'] = pd.to_datetime(orders['buy_ts'], format="%Y-%m-%d %H:%M:%S")


# In[12]:


orders.info()


# ## Verificación del dataframe 'costs'

# In[13]:


costs.info()


# In[14]:


costs


# In[15]:


duplicated_rows = costs[costs.duplicated()]

print("Filas duplicadas:")
print(duplicated_rows.count())


# In[16]:


costs['dt'] = pd.to_datetime(costs['dt'], format="%Y-%m-%d")


# In[17]:


costs.info()



# # Informes y métricas

# ## Visitas

# In[18]:


visits['year'] = visits['start_ts'].dt.year
visits['month'] = visits['start_ts'].dt.month
visits['week'] = visits['start_ts'].dt.week
visits['day'] = visits['start_ts'].dt.day


# In[19]:


visits_per_user = visits.groupby(['year', 'month', 'week', 'day']).agg(n_unique_ids=('uid', 'nunique')).reset_index()

visits_per_user


# ### Usuarios por año

# In[20]:


users_per_year = visits_per_user.groupby('year')['n_unique_ids'].sum().reset_index()

users_per_year


# In[21]:


users_per_year.plot(x='year', y='n_unique_ids', title='Usuarios por año', xlabel='Año',
                     ylabel='Cantidad de usuarios', figsize=[10,10], kind='bar')


# In[22]:


users_per_year['n_unique_ids'].describe()


# ### Usuarios por mes

# In[23]:


users_per_month = visits_per_user.groupby(['year', 'month'])['n_unique_ids'].sum().reset_index()

users_per_month


# In[24]:


users_per_month['year_month'] =users_per_month['year'].astype(str) + '_' + users_per_month['month'].astype(str)


# In[25]:


users_per_month.plot(x='year_month', y='n_unique_ids', title='Usuarios por mes', xlabel='Mes',
                     ylabel='Cantidad de usuarios', figsize=[10,10], kind='bar')


# In[26]:


users_per_month['n_unique_ids'].describe()


# Los meses con mayor cantidad de usuarios en 2017 son son:
#     
#     Octubre 35134
#     
#     Noviembre 39869
#     
#     Diciembre  38470
# 
# 
# Los meses con mayor cantidad de usuarios en 2018 son son:
# 
#     Enero	34002
#     Febrero	34080
#     Marzo	32633


# ### Usuarios por semana

# In[27]:


users_per_week = visits_per_user.groupby(['year','month', 'week'])['n_unique_ids'].sum().reset_index()

users_per_week['year_month_week'] = users_per_week['year'].astype(str) + '_' + users_per_week['month'].astype(str)+ '_' + users_per_week['week'].astype(str)

users_per_week


# In[28]:


users_per_week.plot(x='year_month_week', y='n_unique_ids', title='Usuarios por semana', xlabel='Semana',
                     ylabel='Cantidad de usuarios', figsize=[10,10], kind='bar')


# In[29]:


users_per_week['n_unique_ids'].describe()


# In[30]:


weeks_major_users = users_per_week[users_per_week['n_unique_ids'] >= 7681.25].groupby(['year', 'month', 'week'])['n_unique_ids'].sum().reset_index()

print('Las semanas con mayor número de usuarios son:')
print(weeks_major_users.sort_values(by='n_unique_ids', ascending=False))



# ### Usuarios por día

# In[31]:


users_per_day = visits_per_user.groupby(['year','month', 'day'])['n_unique_ids'].sum().reset_index()

users_per_day['year_month_day'] = users_per_day['year'].astype(str) + '_' + users_per_day['month'].astype(str)+ '_' + users_per_day['day'].astype(str)

users_per_day


# In[32]:


users_per_day.plot(x='year_month_day', y='n_unique_ids', title='Usuarios por día del mes', xlabel='Día del mes',
                     ylabel='Cantidad de usuarios', figsize=[20,10], kind='line', rot=90)


# In[33]:


users_per_day['n_unique_ids'].describe()


# In[34]:


days_major_users = users_per_day[users_per_day['n_unique_ids'] >= 1173].groupby('year_month_day')['n_unique_ids'].sum().reset_index()

print('Los días del mes con mayor número de usuarios son:')
print(days_major_users.sort_values(by='n_unique_ids', ascending=False))



# ### ¿Cuántas sesiones hay por día? (Un/a usuario/a puede tener más de una sesión).
# 

# In[35]:


sess_per_day = visits.groupby(['day', 'uid'])['start_ts'].count().reset_index()

sess_per_day


# In[36]:


sess_user_day = sess_per_day.groupby(['day'])['start_ts'].mean().reset_index()

sess_user_day = sess_user_day.rename(columns={'start_ts': 'mean_sess_per_day'})

sess_user_day


# In[37]:


sess_user_day.plot(x='day', y='mean_sess_per_day', title='Sesiones promedio por día del mes', xlabel='Día del mes',
                     ylabel='Promedio de sesiones', figsize=[10,5], kind='bar')


# In[38]:


sess_user_day['mean_sess_per_day'].describe()


# In[39]:


print(sess_user_day['mean_sess_per_day'].min())
print(sess_user_day['mean_sess_per_day'].max())


# En promedio cada usuario tiene 1.13 sesiones por día. 
# 
# El día con menos sesiones es el 3 con 1.1122 sesiones.
# 
# El día con el mayor número de sesiones es el 24 con 1.1706 sesiones.


# ### ¿Cuál es la duración de cada sesión?
# 

# In[40]:


visits['sess_duration'] = (visits['end_ts'] - visits['start_ts']).dt.total_seconds()


# In[41]:


plt.figure(figsize=(10, 10))
plt.hist(visits['sess_duration'], bins=1000, alpha=0.7, color='blue')
plt.title('Histograma de duración de las sesiones')
plt.xlabel('Duración de la sesión (en segundos)')
plt.ylabel('Frequency')
plt.show()


# In[42]:


print('El promedio de duración de las sesiones por usuario es de:', visits['sess_duration'].mean(), 'segundos.')
print()
print('La mediana en la duración de las sesiones por usuario es de:', visits['sess_duration'].median(), 'segundos.')
print()
print('La moda en la duración de las sesiones por usuario es de:', visits['sess_duration'].mode(), 'segundos.')


# 
#  

# ### Frecuencia con la que los usuarios y las usuarias regresan

# In[43]:


first_visit_date = visits.groupby('uid')['start_ts'].min()
first_visit_date.name = 'primer_visita'

visits = visits.join(first_visit_date, on='uid')

visits['mes_primer_visita'] = visits['primer_visita'].astype('datetime64[M]')


# In[44]:


last_visit_date = visits.groupby('uid')['start_ts'].max()
last_visit_date.name = 'ultima_visita'

visits = visits.join(last_visit_date, on='uid')

visits['mes_ultima_visita'] = visits['ultima_visita'].astype('datetime64[M]')


# In[45]:


visits['mes_primer_visita'] = pd.to_datetime(visits['mes_primer_visita']).dt.to_period('M')
visits['mes_ultima_visita'] = pd.to_datetime(visits['mes_ultima_visita']).dt.to_period('M')

visits['age_months'] = (visits['mes_ultima_visita'] - visits['mes_primer_visita']).apply(lambda x: x.n)


# In[46]:


visits_full = visits.pivot_table(index='mes_primer_visita',
              columns='age_months',
              values='uid',
              aggfunc='nunique')

print(visits_full)


# In[47]:


plt.figure(figsize=(13, 9))
plt.title('Frecuencia con la que los usuarios y las usuarias regresan')
sns.heatmap(visits_full, annot=True, fmt='.1f', linewidths=1, linecolor='gray')


# In[48]:


retention = pd.DataFrame()
for col in visits_full.columns:
    retention = pd.concat([retention, visits_full[col]/visits_full[0]], axis=1)
retention.columns = visits_full.columns
retention.index = [str(x)[0:10] for x in retention.index]
plt.figure(figsize=(13, 9))
sns.heatmap(retention, annot=True, fmt='.1%', linewidths=1, linecolor='grey',  vmax=0.1, cbar_kws= {'orientation': 'horizontal'} 
        ).set(title = 'Retention Rate')
plt.show()
print('{:.1%}'.format(retention[1].mean()))


# In[49]:


freq = visits.groupby(['age_months']).agg(n_unique_ids=('uid', 'nunique')).reset_index()

freq


# In[50]:


freq['percent'] = (freq['n_unique_ids'] / freq['n_unique_ids'].sum())*100

freq


# ## Ventas

# ### ¿Cuándo empieza a comprar la gente?

# 

# Aquí empieza mi intervención

# In[51]:


first_visit_date = visits.groupby('uid')['start_ts'].min().reset_index()
first_visit_date.columns = ['uid', 'first_visit_date']

first_buy_date = orders.groupby('uid')['buy_ts'].min().reset_index()
first_buy_date.columns = ['uid', 'first_buy_date']

ventas = pd.merge(first_visit_date, first_buy_date, on='uid')

ventas['first_order_dt'] = ventas['first_buy_date'].dt.date
ventas['first_session_dt'] = ventas['first_visit_date'].dt.date


ventas['days_to_first_purchase'] = ((ventas['first_buy_date'] - 
                                     ventas['first_visit_date']) / np.timedelta64(1,'D')).astype('int')  


ventas.groupby('days_to_first_purchase')['uid'].count().reset_index()

ventas['days_to_first_purchase'].value_counts().sort_index(ascending=True)


(ventas['days_to_first_purchase'].plot(kind='hist',bins=200, figsize=(12,7))
                                 .set(title = 'Time from visit to order', 
                                      xlabel = 'Days', 
                                      ylabel = 'Frequency'))
plt.xlim(0,100)
plt.show()


# In[52]:


ventas['first_buy_month'] = ventas['first_order_dt'].values.astype('datetime64[M]')

cohort_sizes = ventas.groupby('first_buy_month').agg({'uid': 'nunique'}).reset_index()
cohort_sizes.rename(columns={'uid': 'n_buyers'}, inplace=True)

cohorts = pd.merge(orders, ventas, how='inner', on='uid')
cohorts['buy_dt'] = cohorts['buy_ts'].dt.date
cohorts['order_month'] = cohorts['buy_dt'].values.astype('datetime64[M]')


# In[53]:


cohorts = cohorts\
.groupby(['first_buy_month', 'order_month'])\
.agg({'revenue': 'count'}).reset_index()

cohorts['age_month'] = ((cohorts['order_month'] - cohorts['first_buy_month']) / np.timedelta64(1,'M')).round()

cohorts


# In[54]:


cohorts.columns = ['first_buy_month', 'order_month', 'n_orders', 'age_month']
cohorts_report = pd.merge(cohort_sizes, cohorts, on = 'first_buy_month')
cohorts_report['orders_per_buyer'] = cohorts_report['n_orders'] / cohorts_report['n_buyers']

ordenes_acumuladas = cohorts_report.pivot_table(
    index='first_buy_month', 
    columns='age_month', 
    values='n_orders').cumsum(axis=1)


plt.figure(figsize=(13, 9))
sns.heatmap(ordenes_acumuladas, annot=True, fmt='.1f', linewidths=1, linecolor='gray')


# In[55]:


cohorts_ltv = cohorts_report.pivot_table(
    index='first_buy_month', 
    columns='age_month', 
    values='orders_per_buyer', 
    aggfunc='sum').cumsum(axis=1)

cohorts_ltv.round(2).fillna('')

plt.figure(figsize=(13, 9))
sns.heatmap(cohorts_ltv, annot=True, fmt='.1f', linewidths=1, linecolor='gray')


# Aquí termina mi intervención

# In[56]:


sales = visits.merge(orders, on='uid', how='inner')

compradores = sales[['device', 'end_ts', 'source_id', 'start_ts', 'uid', 'buy_ts', 'revenue']]
compradores = compradores.drop_duplicates(subset=['uid'])


# In[57]:


first_visit_date = visits.groupby('uid')['start_ts'].min().reset_index()
first_visit_date.columns = ['uid', 'first_visit_date']

first_buy_date = orders.groupby('uid')['buy_ts'].min().reset_index()
first_buy_date.columns = ['uid', 'first_buy_date']


# In[58]:


compradores = compradores.merge(first_visit_date, on='uid')
compradores = compradores.merge(first_buy_date, on='uid')


# In[59]:


compradores['months_to_first_buy'] = (compradores['first_buy_date'].dt.year - compradores['first_visit_date'].dt.year) * 12 + \
                                (compradores['first_buy_date'].dt.month - compradores['first_visit_date'].dt.month)

compradores.loc[compradores['months_to_first_buy'] < 0, 'months_to_first_buy'] = 0

compradores['cohort_month'] = compradores['months_to_first_buy']


# In[60]:


compradores['cohort_month'].value_counts().sort_index(ascending=True)


# In[61]:


cohort_counts = compradores['cohort_month'].value_counts().sort_index(ascending=True)

for cohort, count in cohort_counts.items():
    print(f"Total de usuarios en el cohorte {cohort} hacen pedidos a los {cohort} : {count}")


# In[62]:


total_users = compradores.shape[0]
cohort_percentages = cohort_counts / total_users * 100

for cohort, percentage in cohort_percentages.items():
    print(f"{percentage:.2f}% usuarios compraron a los {cohort} meses de que se registraron.")


# El 82.38% de los usuarios hicieron su primera antes de que se cumploera el primer mes después de haberse registrado.
# 
# Más del 88% de los usuarios hacen su primera compra hasta un mes después de haberse registrado.

# In[63]:


compradores['days_to_first_buy'] = (compradores['first_buy_date'] - compradores['first_visit_date']).dt.days

compradores


# In[64]:


(compradores['days_to_first_buy'].plot(kind='hist',bins=100, figsize=(12,7))
                             .set(title = 'Time from visit to order', 
                                  xlabel = 'Days', 
                                  ylabel = 'Frequency'))


# In[65]:


cohort_compradores = compradores.groupby('days_to_first_buy')['uid'].count().reset_index()

cohort_compradores.sort_values(by='uid', ascending=False).head(20)


# In[66]:


print(compradores['days_to_first_buy'].median())
print(compradores['days_to_first_buy'].mean())
print()
print(compradores['days_to_first_buy'].describe())


# En promedio, los usuarios hacen su primer compra a los 17 días después de su registro.	
# 
# 26363 usuarios hacen su primer compra antes de que se cumpla el primer mes desde su registro.

# In[67]:


n_buys = compradores.groupby(['cohort_month', 'months_to_first_buy'])['uid'].count().reset_index()

print('El total de compras por cohorte es de:')
print(n_buys)


# In[68]:


pivot_n_buys = n_buys.pivot_table(index='months_to_first_buy',
              columns='cohort_month',
              values='uid',
              aggfunc='sum').cumsum(axis=1)

plt.figure(figsize=(13, 9))
plt.title('Compras por cohorte')
sns.heatmap(pivot_n_buys, annot=True, fmt='.1f', linewidths=1, linecolor='gray')



# ### ¿Cuántos pedidos hacen durante un día?

# In[69]:


pedidos_por_dia = compradores.groupby(['cohort_month', 'buy_ts']).size().reset_index(name='num_pedidos')

dias_por_cohorte = pedidos_por_dia.groupby('cohort_month')['buy_ts'].nunique().reset_index(name='num_dias')

total_pedidos_por_cohorte = compradores.groupby('cohort_month')['uid'].count().reset_index(name='total_pedidos')

pedidos_cohorte = pd.merge(total_pedidos_por_cohorte, dias_por_cohorte, on='cohort_month')

pedidos_cohorte['promedio_pedidos_por_dia'] = pedidos_cohorte['total_pedidos'] / pedidos_cohorte['num_dias']

print('Los usuarios de cada cohorte tienen un promedio de compras mensuales de:')
print(pedidos_cohorte)


# ### ¿Cuál es el tamaño promedio de compra? (por cohorte)

# In[70]:


print(compradores['cohort_month'].nunique())


# In[71]:


sales_mean = compradores.groupby('cohort_month')['revenue'].mean().reset_index()


# In[72]:


sales_mean


# In[73]:


sales_mean['revenue'].describe()


# En promedio cada compra es de $4.8.-
# 
# Los cohortes con el promedio más altos son 1, 4, 7, 10 y 11.

# In[74]:


compradores.head(2)


# In[75]:


median_purchase = compradores.groupby(['months_to_first_buy','cohort_month'])['revenue'].median().reset_index()
median_purchase['age_month'] = ((median_purchase['cohort_month'] - median_purchase['months_to_first_buy']) / np.timedelta64(1,'M')).round()

median_purchase



# ### ¿Cuánto dinero traen? (LTV)

# In[ ]:


sales_ltv = sales.groupby(['cohort_month']).agg({'uid': 'nunique', 'revenue': 'sum'}).reset_index()


# In[ ]:


sales_ltv['ltv'] = sales_ltv['revenue'] / sales_ltv['uid']


# In[ ]:


sales_ltv


# In[ ]:


sales_ltv['ltv'].describe()


# En promedio cada cohorte aporta $238.15.-
# 
# La cohorte que aporta menos es la 9 ($25.09).
# 
# La cohorte que aporta más es la 7 ($1006.38)

# ## Marketing

# ### ¿Cuánto dinero se gastó? (Total/por fuente de adquisición/a lo largo del tiempo)

# In[ ]:


invest_per_source = costs.groupby(['source_id'])['costs'].sum().reset_index()

print('La cantidad de dinero invertido en cada fuente de adquisición es de:')
print(invest_per_source.sort_values(by='costs', ascending=False))


# In[ ]:


invest_per_source['costs'].describe()


# Las fuentes en las que se han tenido más costos son:
# 
#     source_id        costo
#     
#         3           141321.63
# 
#         4           61073.60



# ### ¿Cuál fue el costo de adquisición de clientes de cada una de las fuentes?

# In[ ]:


unit_economy = sales[['source_id', 'uid', 'revenue']]

unit_economy


# In[ ]:


unit_economy = unit_economy.groupby(['source_id']).agg({'uid':'count', 'revenue':'sum'}).reset_index()

unit_economy


# In[ ]:


costs_ = costs.groupby('source_id')['costs'].sum().reset_index()

costs_


# In[ ]:


unit_economy_ = unit_economy.merge(costs_, on = 'source_id', how = 'left')

unit_economy_


# In[ ]:


unit_economy_['cac'] = unit_economy_['costs'] / unit_economy_['uid']

unit_economy_['ltv'] = unit_economy_['revenue'] / unit_economy_['uid']

unit_economy_


# ### 		¿Cuán rentables eran las inversiones? (ROMI)

# In[ ]:


unit_economy_['romi'] = unit_economy_['ltv'] / unit_economy_['cac']

unit_economy_


# De acuerdo a los resultados, al obtener la rentabilidad de las inversiones, las fuentes a las que se le debe dar preferencia es a aquellas cuyo source_id es 1 y 2, pues el ROMI de ambas es 110.31 y 61.63 respectivamente.
