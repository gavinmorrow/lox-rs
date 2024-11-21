use std::{env, fs, io};

fn main() {
    let mut args = env::args();

    match args.nth(1) {
        Some(path) => match fs::read_to_string(path) {
            Ok(source) => run(source),
            Err(err) => panic!("Error reading file: {err}"),
        },
        None => {
            // run repl
            eprint!("> ");
            while let Some(Ok(line)) = io::stdin().lines().next() {
                run(line);
                eprint!("\n> ");
            }
            eprintln!("Goodbye! o/");
        }
    }
}

fn run(source: String) {
    let tokens = scanner::scan(source);
    dbg!(&tokens);

    let ast = parser::Parser::new(tokens).parse();
    dbg!(&ast);

    if let Ok(ast) = ast {
        let res = interperter::interpert(ast);
        dbg!(res);
    }
}

mod scanner {
    pub fn scan(source: String) -> Vec<Token> {
        let mut tokens = vec![];

        let mut source = source.char_indices().peekable();
        while let Some((pos, char)) = source.next() {
            use TokenType::*;

            let mut equals_variant = |no_eq: TokenType, eq: TokenType| -> TokenType {
                match source.next_if(|(_, c)| *c == '=') {
                    Some(_) => eq,
                    None => no_eq,
                }
            };

            let data = match char {
                '(' => LeftParen,
                ')' => RightParen,
                '{' => LeftBrace,
                '}' => RightBrace,
                ',' => Comma,
                '.' => Dot,
                '-' => Minus,
                '+' => Plus,
                ';' => Semicolon,
                '*' => Star,

                '!' => equals_variant(Bang, BangEqual),
                '=' => equals_variant(Equal, EqualEqual),
                '<' => equals_variant(Less, LessEqual),
                '>' => equals_variant(Greater, GreaterEqual),

                '/' => match source.next_if(|(_, c)| *c == '/') {
                    Some(_) => {
                        // consume until newline
                        while source.next_if(|(_, c)| *c != '\n').is_some() {}
                        continue;
                    }
                    None => Slash,
                },
                '"' => 'string: {
                    let mut chars = vec![];
                    loop {
                        let next = source.next();
                        if let Some((_, char)) = next {
                            if char == '"' {
                                break;
                            }
                            chars.push(char);
                        } else {
                            break 'string Error(ScanError::UnterminatedStringLiteral);
                        }
                    }
                    String(std::string::String::from_iter(chars))
                }
                digit if digit.is_ascii_digit() => {
                    let mut chars = vec![digit];
                    while let Some((_, digit)) = source.next_if(|(_, c)| c.is_ascii_digit()) {
                        chars.push(digit);
                    }
                    // todo: float literals
                    // requires 2 chars of lookahead
                    // if let Some((_, dot)) = source.next_if(|(_, c)| *c == '.') {
                    //     chars.push(dot);
                    //     while let Some((_, digit)) = source.next_if(|(_, c)| c.is_ascii_digit()) {
                    //         chars.push(digit);
                    //     }
                    // }

                    let num = std::string::String::from_iter(chars)
                        .parse::<f64>()
                        .expect("parsed number literal should be valid");
                    Number(num)
                }

                whitespace if whitespace.is_whitespace() => continue,
                ident if ident.is_alphabetic() || ident == '_' => {
                    let mut ident = vec![ident];
                    while let Some((_, c)) =
                        source.next_if(|(_, c)| c.is_alphanumeric() || *c == '_')
                    {
                        ident.push(c);
                    }
                    let ident = std::string::String::from_iter(ident);
                    match ident.as_str() {
                        "and" => And,
                        "class" => Class,
                        "else" => Else,
                        "false" => False,
                        "for" => For,
                        "fun" => Fun,
                        "if" => If,
                        "nil" => Nil,
                        "or" => Or,
                        "print" => Print,
                        "return" => Return,
                        "super" => Super,
                        "this" => This,
                        "true" => True,
                        "var" => Var,
                        "while" => While,
                        ident => Identifier(ident.to_string()),
                    }
                }
                char => Error(ScanError::UnexpectedChar(char)),
            };

            let token = Token { data, pos };

            tokens.push(token);
        }

        tokens
    }

    #[derive(Clone, Debug)]
    pub struct Token {
        pub data: TokenType,
        pub pos: usize,
    }

    #[derive(Clone, Debug)]
    pub enum TokenType {
        // Single-character tokens.
        LeftParen,
        RightParen,
        LeftBrace,
        RightBrace,
        Comma,
        Dot,
        Minus,
        Plus,
        Semicolon,
        Slash,
        Star,

        // One or two character tokens.
        Bang,
        BangEqual,
        Equal,
        EqualEqual,
        Greater,
        GreaterEqual,
        Less,
        LessEqual,

        // Literals.
        Identifier(String),
        String(String),
        Number(f64),

        // Keywords.
        And,
        Class,
        Else,
        False,
        Fun,
        For,
        If,
        Nil,
        Or,
        Print,
        Return,
        Super,
        This,
        True,
        Var,
        While,

        Error(ScanError),
    }

    #[derive(Copy, Clone, Debug)]
    pub enum ScanError {
        UnexpectedChar(char),
        UnterminatedStringLiteral,
    }
}

mod parser {
    use std::iter::Peekable;

    use crate::{
        ast::{
            Ast, Binary, Comparison, ComparisonOperator, Declaration, Equality, EqualityOperator,
            Expr, Factor, FactorOperator, Primary, Stmt, Term, TermOperator, Unary, UnaryOperator,
            VarDecl,
        },
        scanner::{Token, TokenType},
    };

    pub struct Parser<T> {
        tokens: T,
    }

    impl<T: Iterator<Item = Token>> Parser<Peekable<T>> {
        pub fn new<I: IntoIterator<IntoIter = T>>(tokens: I) -> Self {
            let tokens = tokens.into_iter().peekable();
            Parser { tokens }
        }

        pub fn parse(&mut self) -> Result<Ast, ParseError> {
            let mut statements = vec![];

            while self.tokens.peek().is_some() {
                statements.push(self.declaration()?)
            }

            Ok(statements)
        }

        fn declaration(&mut self) -> Result<Declaration, ParseError> {
            if self.matches(|t| matches!(t.data, TokenType::Var)) {
                Ok(Declaration::VarDecl(self.var_decl()?))
            } else {
                Ok(Declaration::Statement(self.statement()?))
            }
        }

        fn var_decl(&mut self) -> Result<VarDecl, ParseError> {
            let token = self.consume(
                |t| matches!(&t.data, TokenType::Identifier(_)),
                ParseErrorType::ExpectedVarName,
            )?;
            let TokenType::Identifier(name) = token.data else {
                unreachable!();
            };

            let initializer = if self.matches(|t| matches!(&t.data, TokenType::Equal)) {
                Some(self.expression()?)
            } else {
                None
            };

            Ok(VarDecl { name, initializer })
        }

        fn statement(&mut self) -> Result<Stmt, ParseError> {
            let stmt = if self.matches(|t| matches!(t.data, TokenType::Print)) {
                Stmt::Print(self.expression()?)
            } else {
                Stmt::Expression(self.expression()?)
            };

            self.consume(
                |t| matches!(t.data, TokenType::Semicolon),
                ParseErrorType::ExpectedSemicolon,
            )?;
            Ok(stmt)
        }

        fn binary<Operand, Operator>(
            &mut self,
            mut operand: impl FnMut(&mut Self) -> Result<Operand, ParseError>,
            operator: impl Fn(&TokenType) -> Option<Operator>,
        ) -> Result<Binary<Operand, Operator>, ParseError> {
            let lhs = operand(self)?;
            let mut expr = Binary { lhs, rhs: vec![] };

            // Allow for parsing a sequence (eg 1 + 2 + 3)
            while let Some(rhs) = self
                .tokens
                .peek()
                // add operator
                .and_then(|t| operator(&t.data))
                .map(|op| {
                    // advance iterator if `operator()` matched
                    self.tokens.next();
                    op
                })
                // add rhs operand
                .map(|op| operand(self).map(|operand| (op, operand)))
                .transpose()?
            {
                expr.rhs.push(rhs);
            }

            Ok(expr)
        }

        fn expression(&mut self) -> Result<Expr, ParseError> {
            Ok(Expr::Equality(self.equality()?))
        }

        fn equality(&mut self) -> Result<Equality, ParseError> {
            self.binary(Self::comparison, |token| match token {
                TokenType::BangEqual => Some(EqualityOperator::NotEqual),
                TokenType::EqualEqual => Some(EqualityOperator::Equal),
                _ => None,
            })
        }

        fn comparison(&mut self) -> Result<Comparison, ParseError> {
            self.binary(Self::term, |token| match token {
                TokenType::Greater => Some(ComparisonOperator::Greater),
                TokenType::GreaterEqual => Some(ComparisonOperator::GreaterEqual),
                TokenType::Less => Some(ComparisonOperator::Less),
                TokenType::LessEqual => Some(ComparisonOperator::LessEqual),
                _ => None,
            })
        }

        fn term(&mut self) -> Result<Term, ParseError> {
            self.binary(Self::factor, |token| match token {
                TokenType::Minus => Some(TermOperator::Subtract),
                TokenType::Plus => Some(TermOperator::Add),
                _ => None,
            })
        }

        fn factor(&mut self) -> Result<Factor, ParseError> {
            self.binary(Self::unary, |token| match token {
                TokenType::Slash => Some(FactorOperator::Divide),
                TokenType::Star => Some(FactorOperator::Multiply),
                _ => None,
            })
        }

        fn unary(&mut self) -> Result<Unary, ParseError> {
            let operator = self
                .tokens
                .peek()
                .and_then(|token| match &token.data {
                    TokenType::Bang => Some(UnaryOperator::Not),
                    TokenType::Minus => Some(UnaryOperator::Negate),
                    _ => None,
                })
                .map(|op| {
                    self.tokens.next();
                    op
                });

            if let Some(operator) = operator {
                let unary = Box::new(self.unary()?);
                Ok(Unary::Unary { operator, unary })
            } else {
                Ok(Unary::Primary(self.primary()?))
            }
        }

        fn primary(&mut self) -> Result<Primary, ParseError> {
            let Some(next_token) = self.tokens.next() else {
                return Err(ParseError::new(ParseErrorType::ExpectedPrimary, None));
            };

            use Primary::{False, Identifier, Nil, Number, String, True};
            match &next_token.data {
                TokenType::False => Ok(False),
                TokenType::True => Ok(True),
                TokenType::Nil => Ok(Nil),

                TokenType::Number(n) => Ok(Number(*n)),
                TokenType::String(s) => Ok(String(s.clone())),
                TokenType::Identifier(name) => Ok(Identifier(name.clone())),

                TokenType::LeftParen => {
                    let expr = self.expression()?;
                    self.consume(
                        |t| matches!(t.data, TokenType::RightParen),
                        ParseErrorType::ExpectedRightParen,
                    )?;
                    Ok(Primary::Grouping(Box::new(expr)))
                }

                _ => Err(ParseError::new(
                    ParseErrorType::ExpectedPrimary,
                    Some(&next_token),
                )),
            }
        }

        fn matches(&mut self, f: impl Fn(&Token) -> bool) -> bool {
            self.tokens.next_if(f).is_some()
        }

        fn consume(
            &mut self,
            f: impl Fn(&Token) -> bool,
            err_type: ParseErrorType,
        ) -> Result<Token, ParseError> {
            self.tokens
                .next_if(f)
                .ok_or(ParseError::new(err_type, self.tokens.peek()))
        }
    }

    #[derive(Debug)]
    pub struct ParseError {
        error: ParseErrorType,
        location: Option<Token>,
    }
    impl ParseError {
        // Take `&Token` rather than just `Token` so that it's easier to use
        // `self.tokens.peek()` w/ it.
        pub fn new(error: ParseErrorType, location: Option<&Token>) -> Self {
            let location = location.map(|t| t.clone());
            ParseError { error, location }
        }
    }

    #[derive(Debug)]
    pub enum ParseErrorType {
        UnexpectedEof,
        ExpectedPrimary,
        ExpectedRightParen,
        ExpectedSemicolon,
        ExpectedVarName,
    }
}

mod interperter {
    use std::collections::HashMap;

    use crate::ast::{
        Ast, Binary, ComparisonOperator, EqualityOperator, Expr, FactorOperator, Primary, Stmt,
        TermOperator, Unary, UnaryOperator,
    };

    pub fn interpert(ast: Ast) -> Result<(), Error> {
        for stmt in ast {
            // match stmt {
            //     Stmt::Expression(expr) => {
            //         expr.evaluate(env)?;
            //     }
            //     Stmt::Print(expr) => {
            //         let value = evaluate(expr)?;
            //         println!("{value}");
            //     }
            // }
        }
        Ok(())
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        Nil,
        Boolean(bool),
        Number(f64),
        String(String),
    }

    impl Value {
        fn is_truthy(&self) -> bool {
            use Value::{Boolean, Nil};
            match self {
                Nil | Boolean(false) => false,
                _ => true,
            }
        }
    }

    impl std::fmt::Display for Value {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let str = match self {
                Value::Nil => "nil",
                Value::Boolean(b) => &b.to_string(),
                Value::Number(n) => &n.to_string(),
                Value::String(s) => s,
            };
            f.write_str(str)
        }
    }

    trait BinaryOperator {
        fn apply(&self, a: Value, b: Value) -> Result<Value, Error>;
    }

    impl BinaryOperator for EqualityOperator {
        fn apply(&self, a: Value, b: Value) -> Result<Value, Error> {
            match self {
                EqualityOperator::Equal => Ok(Value::Boolean(a == b)),
                EqualityOperator::NotEqual => Ok(Value::Boolean(a != b)),
            }
        }
    }

    impl BinaryOperator for ComparisonOperator {
        fn apply(&self, a: Value, b: Value) -> Result<Value, Error> {
            use Value::{Boolean, Number};
            let (Number(a), Number(b)) = (a, b) else {
                return Err(Error::TypeError);
            };
            match self {
                ComparisonOperator::Less => Ok(Boolean(a < b)),
                ComparisonOperator::LessEqual => Ok(Boolean(a <= b)),
                ComparisonOperator::Greater => Ok(Boolean(a > b)),
                ComparisonOperator::GreaterEqual => Ok(Boolean(a >= b)),
            }
        }
    }

    impl BinaryOperator for TermOperator {
        fn apply(&self, a: Value, b: Value) -> Result<Value, Error> {
            use Value::Number;
            let (Number(a), Number(b)) = (a, b) else {
                return Err(Error::TypeError);
            };
            match self {
                TermOperator::Add => Ok(Number(a + b)),
                TermOperator::Subtract => Ok(Number(a - b)),
            }
        }
    }

    impl BinaryOperator for FactorOperator {
        fn apply(&self, a: Value, b: Value) -> Result<Value, Error> {
            use Value::Number;
            let (Number(a), Number(b)) = (a, b) else {
                return Err(Error::TypeError);
            };

            match self {
                FactorOperator::Divide => Ok(Number(a / b)),
                FactorOperator::Multiply => Ok(Number(a * b)),
            }
        }
    }

    struct Environment {
        values: HashMap<String, Value>,
    }

    impl Environment {
        pub fn new() -> Self {
            Environment {
                values: HashMap::new(),
            }
        }

        pub fn define(&mut self, name: impl Into<String>, value: Value) {
            self.values.insert(name.into(), value);
        }

        pub fn get(&self, name: impl AsRef<str>) -> Option<&Value> {
            self.values.get(name.as_ref())
        }
    }

    trait Evaluate {
        fn evaluate(self, env: &mut Environment) -> Result<Value, Error>;
    }

    impl<Operand: Evaluate, Operator: BinaryOperator> Evaluate for Binary<Operand, Operator> {
        fn evaluate(self, env: &mut Environment) -> Result<Value, Error> {
            let mut lhs: Value = self.lhs.evaluate(env)?;

            for (op, rhs) in self.rhs {
                let rhs: Value = rhs.evaluate(env)?;
                lhs = op.apply(lhs, rhs)?;
            }

            Ok(lhs)
        }
    }

    impl Evaluate for Expr {
        fn evaluate(self, env: &mut Environment) -> Result<Value, Error> {
            match self {
                Expr::Equality(equality) => equality.evaluate(env),
            }
        }
    }

    impl Evaluate for Unary {
        fn evaluate(self, env: &mut Environment) -> Result<Value, Error> {
            match self {
                Unary::Unary { operator, unary } => match operator {
                    UnaryOperator::Not => {
                        let value: Value = (*unary).evaluate(env)?;
                        let value = value.is_truthy();
                        Ok(Value::Boolean(!value))
                    }
                    UnaryOperator::Negate => {
                        let Value::Number(n) = (*unary).evaluate(env)? else {
                            return Err(Error::TypeError);
                        };
                        Ok(Value::Number(-n))
                    }
                },
                Unary::Primary(primary) => primary.evaluate(env),
            }
        }
    }

    impl Evaluate for Primary {
        fn evaluate(self, env: &mut Environment) -> Result<Value, Error> {
            use Value::{Boolean, Nil, Number, String};
            match self {
                Primary::Number(n) => Ok(Number(n)),
                Primary::String(s) => Ok(String(s)),
                Primary::True => Ok(Boolean(true)),
                Primary::False => Ok(Boolean(false)),
                Primary::Nil => Ok(Nil),
                Primary::Grouping(expr) => expr.evaluate(env),
                Primary::Identifier(name) => env.get(name).cloned().ok_or(Error::VarNotDefinied),
            }
        }
    }

    #[derive(Debug)]
    pub enum Error {
        TypeError,
        VarNotDefinied,
    }
}

mod ast {
    pub type Ast = Vec<Declaration>;

    #[derive(Debug)]
    pub enum Declaration {
        VarDecl(VarDecl),
        Statement(Stmt),
    }

    #[derive(Debug)]
    pub struct VarDecl {
        pub name: String,
        pub initializer: Option<Expr>,
    }

    #[derive(Debug)]
    pub enum Stmt {
        Expression(Expr),
        Print(Expr),
    }

    #[derive(Debug)]
    pub enum Expr {
        Equality(Equality),
    }

    #[derive(Debug)]
    pub struct Binary<Operand, Operator> {
        pub lhs: Operand,
        pub rhs: Vec<(Operator, Operand)>,
    }

    pub type Equality = Binary<Comparison, EqualityOperator>;
    #[derive(Debug)]
    pub enum EqualityOperator {
        Equal,
        NotEqual,
    }

    pub type Comparison = Binary<Term, ComparisonOperator>;
    #[derive(Debug)]
    pub enum ComparisonOperator {
        Greater,
        GreaterEqual,
        Less,
        LessEqual,
    }

    pub type Term = Binary<Factor, TermOperator>;
    #[derive(Debug)]
    pub enum TermOperator {
        Subtract,
        Add,
    }

    pub type Factor = Binary<Unary, FactorOperator>;
    #[derive(Debug)]
    pub enum FactorOperator {
        Divide,
        Multiply,
    }

    #[derive(Debug)]
    pub enum Unary {
        Unary {
            operator: UnaryOperator,
            unary: Box<Unary>,
        },
        Primary(Primary),
    }
    #[derive(Debug)]
    pub enum UnaryOperator {
        Not,
        Negate,
    }

    #[derive(Debug)]
    pub enum Primary {
        Number(f64),
        String(String),
        True,
        False,
        Nil,
        Grouping(Box<Expr>),
        Identifier(String),
    }
}
