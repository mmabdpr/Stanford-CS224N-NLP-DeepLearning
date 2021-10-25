# Q1.1
# ------------------
import itertools
corpus_words = list(sorted(set(itertools.chain(*corpus))))
num_corpus_words = len(corpus_words)
# ------------------
# Q1.1-Explanation
# ------------------
# Q1.2
# ------------------
M = np.zeros((num_words, num_words), dtype=np.int64)
word2ind = {word:ind for ind, word in enumerate(words)}
for text in corpus:
    windows = np.lib.stride_tricks.sliding_window_view(text, window_size*2+1)
    for window in windows:
        w_t = window[window_size]
        for w_j in window:
            M[word2ind[w_t], word2ind[w_j]] += 1
M = np.maximum(M, M.T)
np.fill_diagonal(M, 0)
# ------------------
# Q1.2-Explanation
# ------------------
# Q1.3
# ------------------
M_reduced = TruncatedSVD(n_components=k, n_iter=n_iters).fit_transform(M)
# ------------------
# Q1.3-Explanation
# ------------------
# Q1.4
# ------------------
x_coords, y_coords = [], []
for word in words:
    coords = M_reduced[word2ind[word]]
    plt.scatter(coords[0], coords[1], marker='x', color='red')
    plt.text(coords[0], coords[1], word, fontsize=9)
plt.show()
# ------------------
# Q1.4-Explanation
# ------------------
# Q1.5
# ------------------
# ------------------
# Q1.5-Explanation
# ecuador and iraq and kuwait are clustered correctly together as they are all countries. oil and barrels and bpd and petroleum are too far from each other considering their meanings. oil and energy have some relationships in their meanings and they are correctly plotted close to each other.
# ------------------
# Q2.1
# ------------------
# ------------------
# Q2.1-Explanation
# kuwait is far from two other countries now. bpd and barrels are far from each other although it seems more reasonable to have them nearer to each other. in this model industry and energy are plotted in really small distance. also ecuador and iraq and petroleum are clustered together. these differences may stem from the difference between the sources (corpra) that these two models are based on. also there is a huge difference in the algorithms used by these models.
# ------------------
# Q2.2
# ------------------
wv_from_bin.most_similar("feet")
# ------------------
# Q2.2-Explanation
# meanings of feet:<br/>
# 1 a unit for measuring distance<br/>
# 2 plural form of foot

# in the above list there are meters, metres, inches (1) and foot, below, floor (2)
# ------------------
# Q2.3
# ------------------
w1 = "right"
w2 = "correct"
w3 = "wrong"
w1_w2_dist = wv_from_bin.distance(w1, w2)
w1_w3_dist = wv_from_bin.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))
# ------------------
# Q2.3-Explanation
# right or wrong are used in the same context more often. it is even possible to replace them with each other without changing surrounding words to make sentence more natural. in contrast to "right" and "wrong", "correct" is used more in formal situations.
# ------------------
# Q2.4
# ------------------
# ------------------
# Q2.4-Explanation
# $x$ = $k$ - $m$ + $w$
# ------------------
# Q2.5
# ------------------
pprint.pprint(wv_from_bin.most_similar(positive=['linux', 'ntfs'], negative=['windows']))
pprint.pprint(wv_from_bin.most_similar(positive=['dell', 'thinkpad'], negative=['lenovo']))
pprint.pprint(wv_from_bin.most_similar(positive=['i3', 'toyota'], negative=['yaris']))
# ------------------
# Q2.5-Explanation
# windows:ntfs :: linux:ext4<br/>
# explanation: ntfs is a filesystem for windows and ext2 is a filesystem for linux

# lenovo:thinkpad :: dell:xps<br/>
# explanation: thinkpad is a laptop series made by lenovo and inspiron is a laptop series made by dell

# yaris:toyota :: i3:bmw
# ------------------
# Q2.6
# ------------------
pprint.pprint(wv_from_bin.most_similar(positive=['australia', 'tehran'], negative=['iran']))
# ------------------
# Q2.6-Explanation
# should be iran:tehran :: australia:canberra
# ------------------
# Q2.7
# ------------------
# ------------------
# Q2.7-Explanation
# when model tries to give words similar to woman it returns words like nurse, pregnant, mother, teacher, homemakder.
# <br/>words like homemaker indicate gender bias. implicitly saying that women are more likely to be a homemaker in a world that men are more likely to be a worker.
# in the second example words like mechanic indicate that some types of jobs are more masculine which again is a type of gender bias
# ------------------
# Q2.8
# ------------------
pprint.pprint(wv_from_bin.most_similar(positive=['europe', 'islam'], negative=['iran']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['iran', 'islam'], negative=['europe']))
# ------------------
# Q2.8-Explanation
# the results show bias between religions and geography (countries) implying that a person who lives in iran has to be a muslim and a person who lives in europe has to be a christian.
# ------------------
# Q2.9
# ------------------
# ------------------
# Q2.9-Explanation
# gender bias is still present in most human cultures. hence some texts may indicate this type of bias which has been learned by this model.<br/>
# the context in which those words are used can also be misleading for the model.<br/>
# to measure these biases we can calculate some gender vectors like "woman - man" and project neutral words on them to see how much gender bias is present in a corpus
# ------------------