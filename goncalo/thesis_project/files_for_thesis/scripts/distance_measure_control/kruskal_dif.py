from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def main():
    # EXP 1 - CHECKING DIFFERENCES WITH KRUSKAL NON-PARAMETRIC: SAD, SSD, AND MEAN CORR FOR REAL AND MASKS GROUPS
    UMCG_PCA_MASKS_SAD = 49.361
    UMCG_PCA_MASKS_SSD = 1.841
    UMCG_PCA_MASKS_MC = 0.071

    NOVEL_UKBB_PCA_MASKS_SAD = 43.441
    NOVEL_UKBB_PCA_MASKS_SSD = 1.996
    NOVEL_UKBB_PCA_MASKS_MC = -0.003

    UMCG_MASKS_SAD = 46.815
    UMCG_MASKS_SSD = 1.855
    UMCG_MASKS_MC = 0.064

    NOVEL_UKBB_MASKS_SAD = 219.497
    NOVEL_UKBB_MASKS_SSD = 1.844
    NOVEL_UKBB_MASKS_MC = -0.021

    UMCG_PCA_REAL_SAD = 168.350
    UMCG_PCA_REAL_SSD = 0.780
    UMCG_PCA_REAL_MC = 0.323

    UMCG_REAL_SAD = 170.458
    UMCG_REAL_SSD = 0.787
    UMCG_REAL_MC = 0.324

    NOVEL_UKBB_PCA_REAL_SAD = 216.694
    NOVEL_UKBB_PCA_REAL_SSD = 1.927
    NOVEL_UKBB_PCA_REAL_MC = 0.007

    NOVEL_UKBB_REAL_SAD = 169.338
    NOVEL_UKBB_REAL_SSD = 0.772
    NOVEL_UKBB_REAL_MC = 0.105

    # real_stat = stats.kruskal(UMCG_PCA_REAL_SAD, UMCG_REAL_SAD, NOVEL_UKBB_PCA_REAL_SAD, NOVEL_UKBB_REAL_SAD)
    # masks_stat = stats.kruskal(UMCG_PCA_MASKS_SAD, NOVEL_UKBB_PCA_MASKS_SAD,UMCG_MASKS_SAD,NOVEL_UKBB_MASKS_SAD)

    # print("Masks stat: ".format(masks_stat))
    # print("Real stat: ".format(real_stat))

    fvalue, pvalue = stats.f_oneway(UMCG_PCA_MASKS_SSD,
                            NOVEL_UKBB_PCA_MASKS_SSD,
                            UMCG_MASKS_SSD,
                            NOVEL_UKBB_MASKS_SSD)
    print("fvalue:{} , and pvalue:{}".format(fvalue, pvalue))

if __name__ == '__main__':
    main()