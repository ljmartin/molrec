import matplotlib.pyplot as plt
plt.style.use('seaborn')

filenames = ['hpo_implicit_als.dat', 'hpo_implicit_bpr.dat',
             'hpo_lightfm_warp.dat', 'hpo_lightfm_bpr.dat']


fig, ax = plt.subplots(6)
fig.set_figheight(12)
fig.set_figwidth(9)

count=0

for filename in filenames:
    file = open(filename, 'r')
    func_evals = list()
    record=False
    for line in file:
        if record:
            func_evals.append(float(line.strip('\n').split()[-1]))
        if 'All' in line:
            record=True

    
    ax[count].plot(func_evals)
    ax[count].set_xlabel('Iterations', fontsize=12)
    ax[count].set_ylabel('Mean rank', fontsize=12)
    ax[count].set_title(filename[:-4], fontsize=15)
    #plt.tight_layout()
    #plt.savefig(filename[:-4]+'.png')
    count+=1

plt.tight_layout()
plt.savefig('all.pdf')
plt.savefig('all.tif')

#plot all in one:
fig, ax = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(7)
for filename in filenames:
    file = open(filename, 'r')
    func_evals = list()
    record=False
    for line in file:
        if record:
            func_evals.append(float(line.strip('\n').split()[-1]))
        if 'All' in line:
            record=True
    ax.plot(func_evals, linewidth=2, label=filename[:-4])

ax.set_xlabel('Iterations', fontsize=12)
ax.set_ylabel('Mean rank', fontsize=12)
ax.set_title('Hyperparameter Optimization', fontsize=15)
ax.set_ylim(0,170)
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, ncol=2, loc='upper center')
plt.tight_layout()
fig.savefig('all_in_one.pdf')
fig.savefig('all_in_one.tif')
plt.close(fig)
