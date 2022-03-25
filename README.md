# kmeans_scratch with finding the optimal K with minimum ratio
1)	Randomly choosing Centroids
2)	finding the distance between centroids and all the data points using Euclidean distance formula
3)	Update the Centroid with the minimum Distance
4)	Repeating the above steps (1-3) for a defined number of iterations
5)	Return center,mean_intra_cluster_distance,mean_inter_cluster_distance
<img width="465" alt="image" src="https://user-images.githubusercontent.com/94094997/159954724-d6be18c4-925e-4d5a-861f-e31b9f5fd716.png">
<img width="469" alt="image" src="https://user-images.githubusercontent.com/94094997/159954757-7ab28277-fb7b-4bde-9887-dc747e498320.png">
# Sample output
<img width="1405" alt="image" src="https://user-images.githubusercontent.com/94094997/160178630-e3b35117-0ed1-4f0c-87d4-92d4fe77b8b6.png">

Note : K -means algorithm does not guarantee to find the optimum.
So when you run the script again and again the optimal ‘k’ value changes each time because of random tensor fake data.

