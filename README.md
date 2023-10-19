There have been various kinds of mesh simplification algorithms. In the assignment, I use the simplification algorithm of [1] which is built on vertex pair contractions and error quadrics. The followings are the summary of the algorithm:
1.	Calculate 𝑄 matrices for each vertex 𝑣𝑖 = [𝑥𝑣 , 𝑦𝑣 , 𝑧𝑣 , 1]𝑇.
Each vertex 𝑣𝑖 is the intersection of a set of planes 𝑝𝑙𝑎𝑛𝑒𝑠(𝑣𝑖) . A plane 𝑝 can be represented as 𝑝 = [𝑎, 𝑏, 𝑐, 𝑑]𝑇 (defined by the equation 𝑎𝑥 + 𝑏𝑦 + 𝑐𝑧 + 𝑑 = 0 where 𝑎2 +
𝑏2 + 𝑐2 = 1). Then 𝑄𝑖 (for vertex 𝑣𝑖) is defined as follows:

 



2.	Select valid vertex pairs.
 
𝑄𝑖 =	∑	𝑝𝑝𝑇
𝑝∈𝑝𝑙𝑎𝑛𝑒𝑠(𝑣𝑖)
 

A vertex pair (𝑣𝑖, 𝑣𝑗) is valid for contraction if either:
(𝑣𝑖, 𝑣𝑗) is an edge, or ||𝑣𝑖, 𝑣𝑗|| < 𝑡 where 𝑡 is a given threshold parameter.
3.	Calculate the optimal contraction vertex 𝑣𝑜𝑝𝑡 which minimizes the error 𝑐𝑜𝑠𝑡𝑖𝑗 for each valid pair  (𝑣𝑖, 𝑣𝑗).
 

First, set 𝑄𝑜𝑝𝑡
 

= 𝑄𝑖
 

+ 𝑄𝑗
 
𝑞11	𝑞12	𝑞13	𝑞14
𝑞12	𝑞22	𝑞23	𝑞24
𝑞13	𝑞32	𝑞33	𝑞34
𝑞14	𝑞24	𝑞34	𝑞44
 
Define the 𝑐𝑜𝑠𝑡𝑖𝑗 as follows:
𝑐𝑜𝑠𝑡𝑖𝑗 = 𝑣𝑇𝑄𝑜𝑝𝑡𝑣

We want to find a solution 𝑣𝑜𝑝𝑡 to minimize the 𝑐𝑜𝑠𝑡𝑖𝑗 . 𝑣𝑜𝑝𝑡 can be calculated as follows:

𝑞11	𝑞12	𝑞13	𝑞14   −1   0
𝑣	= [𝑞12	𝑞22	𝑞23	𝑞24]	0
 
𝑜𝑝𝑡
 
𝑞13
 
𝑞32
 
𝑞33
 
𝑞34	[  ]
 
0	0	0	1	1
 
If the matrix is not invertible. We find the 𝑣𝑜𝑝𝑡 by assigning 𝑣𝑜𝑝𝑡  to be 𝑣𝑖 , 𝑣𝑗 , and the middle point of 𝑣𝑖 and 𝑣𝑗 to see whose 𝑐𝑜𝑠𝑡𝑖𝑗 is the minimum.
4.	Place all the valid pairs in the order of 𝑐𝑜𝑠𝑡𝑖𝑗 with the minimum 𝑐𝑜𝑠𝑡𝑖𝑗 pair on the top.
5.	Remove the minimum 𝑐𝑜𝑠𝑡𝑖𝑗 pair, contract (𝑣𝑖, 𝑣𝑗) to be 𝑣𝑜𝑝𝑡, and update the 𝑐𝑜𝑠𝑡𝑖𝑗
of all valid pairs involving  𝑣𝑖    or  𝑣𝑗.



