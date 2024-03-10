use std::io;
use rand::Rng;
use rayon::prelude::*;

extern crate mpi;

use mpi::traits::*;

pub const NUM_OF_ROWS: usize = 12;
pub const NUM_OF_COLS: usize = 9;
pub const MINIMAX_DEPTH: usize = 5;

pub const SEQ_TO_WIN: usize = 4;
const DEFAULT_EMPTY_CHAR: char = '_';

const FIRST_TEAM: char = 'P';
const SECOND_TEAM: char = 'S';

#[derive(Clone)]
pub struct GameMatrix {
    matrix: [[char; NUM_OF_COLS]; NUM_OF_ROWS],
}


impl GameMatrix {
    pub fn new() -> GameMatrix {
        GameMatrix {
            matrix: [[DEFAULT_EMPTY_CHAR; NUM_OF_COLS]; NUM_OF_ROWS],
        }
    }
    pub fn show(&self) {
        for row in &self.matrix {
            for &value in row {
                print!(" {} ", value);
            }
            println!();
        }
        println!();
        for i in 0..NUM_OF_COLS {
            print!(" {} ", i);
        }
        println!();
    }

    pub fn add_value(&mut self, column: usize, value: char) -> Result<(), String> {
        if column >= NUM_OF_COLS {
            return Err(String::from("Coluna estã fora dos limites"));
        }

        let mut row_index = None;
        for i in (0..NUM_OF_ROWS).rev() {
            if self.matrix[i][column] == DEFAULT_EMPTY_CHAR {
                row_index = Some(i);
                break;
            }
        }
        if let Some(index) = row_index {
            self.matrix[index][column] = value;
            Ok(())
        } else {
            Err(String::from("Coluna está cheia"))
        }
    }

    pub fn check_win(&self, target_char: char) -> bool {
        // Verificar horizontalmente
        for row in &self.matrix {
            let mut count = 0;
            for &value in row {
                if value == target_char {
                    count += 1;
                    if count >= SEQ_TO_WIN {
                        return true;
                    }
                } else {
                    count = 0;
                }
            }
        }

        // Verificar verticalmente
        for col in 0..NUM_OF_COLS {
            let mut count = 0;
            for row in 0..NUM_OF_ROWS {
                if self.matrix[row][col] == target_char {
                    count += 1;
                    if count >= SEQ_TO_WIN {
                        return true;
                    }
                } else {
                    count = 0;
                }
            }
        }

        // Verificar diagonalmente ascendente
        for start_col in 0..NUM_OF_COLS - SEQ_TO_WIN + 1 {
            for start_row in 0..NUM_OF_ROWS - SEQ_TO_WIN + 1 {
                let mut count = 0;
                for i in 0..SEQ_TO_WIN {
                    if self.matrix[start_row + i][start_col + i] == target_char {
                        count += 1;
                        if count >= SEQ_TO_WIN {
                            return true;
                        }
                    } else {
                        count = 0;
                        break;
                    }
                }
            }
        }

        // Verificar diagonalmente descendente
        for start_col in 0..NUM_OF_COLS - SEQ_TO_WIN + 1 {
            for start_row in (SEQ_TO_WIN - 1)..NUM_OF_ROWS {
                let mut count = 0;
                for i in 0..SEQ_TO_WIN {
                    if self.matrix[start_row - i][start_col + i] == target_char {
                        count += 1;
                        if count >= SEQ_TO_WIN {
                            return true;
                        }
                    } else {
                        count = 0;
                        break;
                    }
                }
            }
        }

        false
    }

    pub fn minimax(&mut self, depth: usize, is_maximizing: bool) -> (isize, usize) {
        if depth == 0 || self.check_win(FIRST_TEAM) || self.check_win(SECOND_TEAM) {
            return (self.evaluate_board(), 0);
        }

        let mut moves_scores: Vec<(isize, usize)> = (0..NUM_OF_COLS)
            .into_par_iter()
            .filter_map(|col| {
                let mut new_game = self.clone(); // Clona o estado atual do jogo para explorar essa jogada
                if new_game.add_value(col, if is_maximizing { FIRST_TEAM } else { SECOND_TEAM }).is_ok() {
                    let score = if is_maximizing {
                        let (score, _) = new_game.minimax(depth - 1, false);
                        score
                    } else {
                        let (score, _) = new_game.minimax(depth - 1, true);
                        score
                    };
                    new_game.remove_value(col); // Limpa a jogada explorada
                    Some((score, col))
                } else {
                    None // Coluna estava cheia, movimento não é possível
                }
            })
            .collect();

        // Escolhe o melhor movimento baseado em maximizar ou minimizar a pontuação
        if is_maximizing {
            moves_scores.par_iter().max_by_key(|k| k.0).cloned().unwrap_or((isize::MIN, 0))
        } else {
            moves_scores.par_iter().min_by_key(|k| k.0).cloned().unwrap_or((isize::MAX, 0))
        }
    }


    // Função para remover o último valor inserido em uma coluna específica
    pub fn remove_value(&mut self, column: usize) {
        for row in 0..NUM_OF_ROWS {
            if self.matrix[row][column] != DEFAULT_EMPTY_CHAR {
                self.matrix[row][column] = DEFAULT_EMPTY_CHAR;
                break;
            }
        }
    }


    pub fn evaluate_board(&self) -> isize {
        let (rows_score, (cols_score, diagonals_score)) = rayon::join(
            // DISPARA MPI
            || self.evaluate_rows(),
            || rayon::join(
                // DISPARA MPI
                || self.evaluate_cols(),
                // DISPARA MPI
                || self.evaluate_diagonals(),
            ),
        );

        rows_score + cols_score + diagonals_score
    }

    // Avalia todas as linhas horizontalmente.
    fn evaluate_rows(&self) -> isize {
        (0..NUM_OF_ROWS)
            .into_par_iter()
            .map(|row| self.evaluate_line(self.matrix[row].to_vec()))
            .sum()
    }

    // Avalia todas as colunas verticalmente.
    fn evaluate_cols(&self) -> isize {
        (0..NUM_OF_COLS)
            .into_par_iter()
            .map(|col| {
                let col_vec: Vec<char> = (0..NUM_OF_ROWS).map(|row| self.matrix[row][col]).collect();
                self.evaluate_line(col_vec)
            })
            .sum()
    }

    // Combina a avaliação de diagonais ascendentes e descendentes.
    fn evaluate_diagonals(&self) -> isize {
        let ascendente = self.evaluate_diagonal_ascendente();
        let descendente = self.evaluate_diagonal_descendente();

        ascendente + descendente
    }


    fn evaluate_line(&self, line: Vec<char>) -> isize {
        let mut score = 0;
        let mut count_p = 0;
        let mut count_s = 0;

        for &value in &line {
            if value == FIRST_TEAM {
                count_p += 1;
                count_s = 0;
            } else if value == SECOND_TEAM {
                count_s += 1;
                count_p = 0;
            } else {
                count_p = 0;
                count_s = 0;
            }

            score += match count_p {
                2 => 10,
                3 => 50,
                4 => 100,
                _ => 0,
            };

            score -= match count_s {
                2 => 10,
                3 => 50,
                4 => 100,
                _ => 0,
            };
        }

        score
    }

    // pub fn evaluate_diagonals(&self) -> isize {
    //     let diagonal_scores: (isize, isize) = rayon::join(
    //         || self.evaluate_diagonal_ascendente(),
    //         || self.evaluate_diagonal_descendente(),
    //     );
    //
    //     diagonal_scores.0 + diagonal_scores.1
    // }

    fn evaluate_diagonal_ascendente(&self) -> isize {
        (0..=(NUM_OF_ROWS - SEQ_TO_WIN)).into_par_iter().flat_map(|row| {
            (0..=(NUM_OF_COLS - SEQ_TO_WIN)).into_par_iter().map(move |col| {
                let mut count_p = 0;
                let mut count_s = 0;
                for i in 0..SEQ_TO_WIN {
                    match self.matrix[row + i][col + i] {
                        FIRST_TEAM => count_p += 1,
                        SECOND_TEAM => count_s += 1,
                        _ => {}
                    }
                }
                self.calculate_score(count_p, count_s)
            })
        }).sum()
    }

    fn evaluate_diagonal_descendente(&self) -> isize {
        let mut score = 0;

        // Avaliar diagonais descendentes
        (0..=(NUM_OF_ROWS - SEQ_TO_WIN)).into_par_iter().flat_map(|row| {
            (0..=(NUM_OF_COLS - SEQ_TO_WIN)).into_par_iter().map(move |col| {
                let mut count_p = 0;
                let mut count_s = 0;
                for i in 0..SEQ_TO_WIN {
                    if let Some(r) = row.checked_sub(i) { // Evita underflow
                        match self.matrix.get(r)?.get(col + i) {
                            Some(&FIRST_TEAM) => count_p += 1,
                            Some(&SECOND_TEAM) => count_s += 1,
                            _ => {}
                        }
                    }
                }
                Some(self.calculate_score(count_p, count_s))
            }).filter_map(|x| x)
        }).sum()
    }

    // Definição do método calculate_score dentro do impl GameMatrix
    fn calculate_score(&self, count_p: usize, count_s: usize) -> isize {
        let mut score = 0;
        if count_p == SEQ_TO_WIN {
            score += 100; // Valor exemplo para vitória
        } else if count_s == SEQ_TO_WIN {
            score -= 100; // Valor exemplo para vitória do oponente
        }
        // Lógica adicional pode ser adicionada aqui
        score
    }
}


fn main() {
    let mut game_matrix = GameMatrix::new();
    let mut actual_team = FIRST_TEAM;
    let mut game_ended: bool = false;
    let mut multiplayer: usize;


    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    if (rank == 0) {
        loop {
            let mut input = String::new();
            println!("0 - Deseja jogar sozinho\n1 - Jogar com amigo", );
            io::stdin().read_line(&mut input)
                .expect("Falha ao ler a linha");
            multiplayer = input.trim().parse()
                .expect("Por favor, digite um número válido");
            if (multiplayer == 0 || multiplayer == 1) {
                break;
            }
        }

        game_matrix.show();

        while game_ended != true {
            if (actual_team == SECOND_TEAM && multiplayer == 0) {
                let message = get_ia_message();
                println!("IA: {}", message);
                let is_maximizing = actual_team == FIRST_TEAM;
                let (_, best_col) = game_matrix.minimax(MINIMAX_DEPTH, is_maximizing);
                println!("A resposta de tudo foi: {}", best_col);
                game_matrix.add_value(best_col, actual_team).expect("Falha ao adicionar valor pela IA");

                if game_matrix.check_win(actual_team) {
                    game_ended = true;
                } else {
                    actual_team = change_to_next_team(actual_team);
                }

                game_matrix.show();
            } else {
                println!("Digite a coluna jogador {}:", actual_team);
                let mut number: usize = 10;

                while number > 8 && number != 42 {
                    let mut input = String::new();
                    io::stdin().read_line(&mut input)
                        .expect("Falha ao ler a linha");
                    number = input.trim().parse()
                        .expect("Por favor, digite um número válido");
                    if number > 8 && number != 42 {
                        println!("Por favor, a maior coluna e 8")
                    }
                }

                if number == 42 {
                    let is_maximizing = actual_team == FIRST_TEAM;
                    let (_, best_col) = game_matrix.minimax(MINIMAX_DEPTH, is_maximizing);
                    println!("A resposta de tudo foi: {}", best_col);
                    game_matrix.add_value(best_col, actual_team).expect("Falha ao adicionar valor pela IA");
                } else {
                    let _ = game_matrix.add_value(number, actual_team).is_ok();
                }

                if game_matrix.check_win(actual_team) {
                    game_ended = true;
                } else {
                    actual_team = change_to_next_team(actual_team);
                }

                game_matrix.show();
            }
        }


        println!("O JOGADOR VENCEDOR FOI {}", actual_team);
    }
}

fn get_ia_message() -> &'static str {
    let messages = [
        "A IA está pensando...",
        "Essa ia ser a minha jogada, e agora...",
        "Por essa eu não esperava, mas posso fazer isso...",
        "Não estrague os meus planos...",
        "Com essa estrategia eu vou ganhar...",
        "Se fosse truco, eu ja tinha ganhado...",
        "Comecou mal e agora parece o comeco...",
        "Está jogando vendado?",
    ];

    let mut rng = rand::thread_rng();
    let random_number = rng.gen_range(0..messages.len());

    return messages[random_number];
}

fn change_to_next_team(actual_team: char) -> char {
    if actual_team == FIRST_TEAM {
        return SECOND_TEAM;
    }
    return FIRST_TEAM;
}
