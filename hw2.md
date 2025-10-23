I by done what leave death,
And aproposely beef the are and sors blate though wat our fort


Without changing anything, the loss obtained was 1.6958 (4,4)
4 layers, 4 heads: step 2000: train loss 1.7648, val loss 1.8857
20 layers, 16 heads: step 2000: train loss 1.7610, val loss 1.8443
10 layers, 16 heads: step 2000: train loss 1.7458, val loss 1.8868
10 layers, 8 heads: step 2000: train loss 1.7375, val loss 1.8834
10 layes, 4 heads: step 2000: train loss 1.7572, val loss 1.9021
100 layers, 16 heads: step 2000: train loss 1.7191, val loss 1.8144


Evaluation metrics: 
General: use Distinct-N to measure generation diversity; Also, since we're training per-character models, I also measure how many words are actually a word in the input text. 
Specific: use perplexity to measure how close the generation is to the training tata. 
Average Perplexity over 10 samples: 6.41
Word overlap: 83/148 = 0.5608
Distinct-1: 0.7041
Distinct-2: 0.9846
Distinct-3: 1.0000
Distinct-4: 1.0000

I used H.P. Lovecraft's fiction from https://www.kaggle.com/datasets/bennijesus/lovecraft-fiction/data, and trained with the same configuration as the shakespeare_char, with 50 layers and 8 heads. I splitted the input.txt into chunks of 10000, 20000, 30000, ..... characters, trained and evaluated separately, and then plotted all metrics. 

The most significant one is "word overlap". When the training data size reaches 40000, the words generated are more likely to be the words that appears in the original text. After 40000 there isn't more significant improvements. Perplexity and Distinct-n are all fluctuating. 

In order to finetune the shakespeare_char model, I'll need to encode the lovecraft data in the same way as the shakespeare data. This can be done by retaining the integer mapping for the characters that are in both dataset, and then add new integer mappings for characters that only appears in the lovecraft dataset. I changed the prepare.py for lovecraft dataset. 

for 10000 size dataset:
step 2020: train loss 1.9491, val loss 1.8974
The output looks like: 
> I gupelle for than piliturting hore royes he bars to the should-hus
> cond the parce of it Her dead eet the paited lister to haver shall shalle of the father are his him.
> Halst gare to die should art with me did of he signe meer.
> 
> LUCIO:
> Not the sonteert of in me verice slay of leff the bridecdes in of in man
> the grover be to enousher and revight are sogh of the togertome,
> The like the hings of by better sillige to to and the cupreas.
> 
> Nost Sepond Norse:
> Here of sirtue and her clouss me on I am th

Increase the data size by using the 310000 size dataset for finetuning: 
step 2020: train loss 2.0417, val loss 2.0431
Output:
> GLOUCESTER:
> I conderer your in within of blient and these of the pirstab'd a bed'dimter act
> again the feator in his many would flarce of in a grien'daintien and with as like.
> The bardelkn sleed the delece --

> HAMY COOLLON:
> Of thy surverech or charnes? We shright Locks as to impery am bart
> For butter sickle of the didhices in of worldsen awited.
> Fall streling in some me canse live and sufferrited are it.
> Land's of son yether far holden you conter to come a did tis.
> Maker your stame well, murse in mamy that a heam of or the mutter
> best unfil brack and as was in to thepped reats. You hose art good that art sule his Kiltined our challow son,
> Samein and in nother 

Now it looks more like Lovecraft. 