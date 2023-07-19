from flask import Flask,request,render_template
import run

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])

def ree():
    if(request.method == 'POST'):
        data = request.form
        #print(data)
        str = data.get('sen')

        #todo
        X, y = run.inter(str)
        ss = []
        for i in range(len(X)):
            ss.append(X[i]+"\t-\t<"+y[i]+">")
        print(str)

        #test
        # X = str.split()
        # ss = []
        # for i in X:
        #     ss.append(i+"\t-\t"+"<yes>")

        return render_template('pro.html', u = ss)
    return render_template('pro.html')


if __name__ == '__main__':
    app.run(debug=True)
