class InterfaceTerminal:
    @staticmethod
    def print_progress_bar(attempt, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (attempt / float(total)))
        filled_length = int(length * attempt // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        return print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
        # if attempt == total:
        #     return print()


