# graph_transcript
You are given a transcript from a lecture record-
ing. Your task is to generate a directed graph based on the sentences in the
transcript.
Each node represents a set of words from a sentence. Nodes are connected
if their sentences share common words, with directed edges labeled by the
shared words.
Your task is to:
• Build a directed graph based on the sentences, where:
– Each node is a collection of words appearing in a sentence.
– Two nodes are connected if their corresponding sentences share
some common words, directed from the earlier sentence to the
latter one.
– The edge is labeled by the shared words, weighted by the number
of shared words.
• Print the number of vertices, the number of edges, and the number of
connected components (for this part you may want to treat the graph
as undirected. You may also want to compute the undirected degree
distribution).
• Plot the weighted in-degree and out-degree distribution (weighted by
edge weights) of the graph.
• Find the strongly connected components (SCCs) in the graph. Draw
the DAGs of SCCs in non-decreasing order of the number of vertices.
• Plot the distribution of the SCCs by size in the number of vertices and
the number of edges.
• Convert the graph to an undirected graph. Compute the bi-connected
components (BCCs) in the graph. Draw the BCCs forest.
Note:
• Punctuation marks, spaces, line breaks, and stop words1 are ignored
when constructing the graph.
1spaCy stop_words.py
4
• You could use Python modules like pypdf2 for extracting texts from
PDF files, NLTK3
, spaCy4 for natural language processing, igraph5
,
NetworkX6 forgraphprocessing,andJavaScriptlibrary2Dforce-graph7
,
3D Force-Directed Graph8. Of course, you are welcome to use any
other libraries of your choice.
• APythonscriptfornaturallanguagepreprocessingisavailableinOneDrive.
YourinputsareaccessibleinOneDrivefoldernamed“Group-[Your_Group_Number]”.
What to submit:
• Your program source code for all tasks.
• A size.txt file indicates the number of vertices, the number of edges,
and the number of connected components.
• A degree-dist.png file plots the undirected degree distribution.
• An in-deg-dist.png and an out-deg-dist.png files plot the weighted de-
gree distributions.
• An scc-dag.png file draws the DAGs of SCCs.
• An scc-size.png file plots the size of SCCs.
• A bcc-forest.png file draws the BCCs forest.
• A short (two-minute) demo.mp4 video file shows the interaction with
your drawings.