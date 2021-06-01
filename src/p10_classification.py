import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/hw1/pokemon.csv')
type1s = list(set(df['type1']))
print("type1s: ", len(type1s), ", ", type1s)
for i in range(len(type1s)):
    type1_name = type1s[i]
    type1_count = len(df[:400].loc[df['type1'] == type1s[i]])
    print("%s: %d" % (type1_name, type1_count))

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

df2 = df[:400].loc[df['type1'] == 'water', ['defense', 'sp_defense']]
defense_data = df2.values[:, 0]
sp_defense_data = df2.values[:, 1]
print(type(defense_data), type(sp_defense_data))
for i in range(len(defense_data)):
    ax1.cla()
    ax1.scatter(defense_data[:i], sp_defense_data[:i])
    ax1.set_xlim(0, 190)
    ax1.set_ylim(0, 170)
    plt.pause(.1)

df_normal = df[:400].loc[df['type1'] == 'normal', ['defense', 'sp_defense']]
defense_data_normal = df_normal.values[:, 0]
sp_defense_data_normal = df_normal.values[:, 1]
print(type(defense_data_normal), type(sp_defense_data_normal))
for i in range(len(defense_data_normal)):
    ax2.cla()
    ax2.scatter(defense_data_normal[:i], sp_defense_data_normal[:i])
    ax2.set_xlim(0, 190)
    ax2.set_ylim(0, 170)
    plt.pause(.1)

plt.ioff()
plt.show()
