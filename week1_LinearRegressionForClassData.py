import statsmodels.api
import numpy

def main():
    (N, X, Y) = read_data()

    results = do_multivariate_regression(N, X, Y)
    print(results.summary())

    effective_variables = get_effective_variables(results)
    print(effective_variables)

def read_data():
    # 1

    # 30명 학생들이 변수(X)[수면 시간 및 기상 시간, 하루 중 나누는 대화의 수와 시간, 운동 (걷기, 앉아있기, 달리기, 서있기), 학생의 위치 정보 (기숙사, 수업, 파티, 운동), 학생 주변에 있었던 사람들, 스트레스 레벨, 식습관] 에 대한 GPA정보(Y)를 기록한 data를 읽어와 저장한다.
    X = []
    Y = []
    N = 0
    
    with open("students.dat") as f:
        definition = f.readline()
        
        
        for line in f :
            temp = []
            value = line.strip().split()
            
            temp.append(float(value[0]))
            temp.append(float(value[1]))
            temp.append(float(value[2]))
            temp.append(float(value[3]))
            temp.append(float(value[4]))
            
            X.append(temp)
            Y.append(float(value[5]))
            
            N += 1
                
    # X must be numpy.array in (30 * 5) shape.
    # Y must be 1-dimensional numpy.array.

    X = numpy.array(X)
    Y = numpy.array(Y)
    
    return (N, X, Y)

def do_multivariate_regression(N, X, Y):
    # 2
    
    # 변수가 여러개인 다중 선형 회귀법을 적용
    
    results = statsmodels.api.OLS(Y, X).fit()

    return results

def get_effective_variables(results):
    eff_vars = []
    # 3
    
    # 다중 선형 회귀법의 결과 중 P>|t| 값이 0.05 인 것만 변수로서의 가치가 유효하고, 나머지는 Y값(GPA)와 무관한 상관관계를 갖으므로 제외한다. 다시 말해, 여러 변수 중 유효한 것을 선발하는 과정이다.
    
    for i in range (len(results.pvalues)) :
        if results.pvalues[i] < 0.05 :
            eff_vars.append('x' + str(i+1))

    return eff_vars

def print_students_data():
    with open("students.dat") as f:
        for line in f:
            print(line)

if __name__ == "__main__":
    main()