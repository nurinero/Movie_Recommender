from flask import Flask,render_template, request
from recommender import recommend_random,recommend_with_NMF
from utils import movies,movie_to_id,id_to_movie
app = Flask(__name__)

@app.route('/')
def hello():
    """
    Returns:
        hello is printed out
    """
    
    return render_template('index.html',name ='Movie Covie Club',movies = movies.title.to_list())


@app.route('/recommend')
def recommendations():
    if request.args['algo']=='Random':
        titles = request.args.getlist('title')
        ratings = request.args.getlist('Ratings')
        
        user_input = dict(zip(titles,ratings))
        print(user_input)

        for keys in user_input:
            user_input[keys] = int(user_input[keys])

        top_Number=int(request.args.getlist('Top_Number')[0])
        print(top_Number)
        
        recs = recommend_random(top_Number) 

        print(request.args)

        return render_template('recommend.html',recs =recs)
     
    elif request.args['algo']=='NMF':
        titles = request.args.getlist('title')
        ratings = request.args.getlist('Ratings')
        top_Number=request.args.getlist('Top_Number')
        
        print(len(titles))
        m_id=movie_to_id(titles)

        user_input = dict(zip(m_id,ratings))
        #print(user_input)

        for keys in user_input:
            user_input[keys] = int(user_input[keys])
        
        print("NEW=====>>",user_input)
        top_Number=int(request.args.getlist('Top_Number')[0])
        print(top_Number)
        recs = recommend_with_NMF("nuri",user_input,top_Number) 
        print(request.args)

        return render_template('recommend.html',recs =recs)
    else:
        return f"Function not defined"







if __name__=='__main__':
    app.run(debug=True,port=5000)