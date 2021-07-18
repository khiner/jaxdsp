// E.g. long_parameter_name => Long parameter name
export function snakeCaseToSentence(name) {
  return name
    ?.split('_')
    .join(' ')
    .replace(/^(.)/, firstLetter => firstLetter.toUpperCase())
}
