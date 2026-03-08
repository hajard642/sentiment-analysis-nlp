import { NextRequest, NextResponse } from 'next/server'

// Mock sentiment analysis function
// In production, this would load the PyTorch model and run inference
function analyzeSentiment(text: string) {
  // Simple heuristic-based sentiment analysis for demonstration
  const positiveWords = ['excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant',
    'perfect', 'outstanding', 'superb', 'incredible', 'masterpiece', 'loved', 'best',
    'great', 'awesome', 'beautiful', 'outstanding', 'stunning', 'brilliant', 'fabulous']
  
  const negativeWords = ['terrible', 'awful', 'horrible', 'disappointing', 'bad',
    'poor', 'dreadful', 'boring', 'waste', 'pathetic', 'hate', 'worst', 'terrible',
    'disappointing', 'awful', 'unwatchable', 'mediocre', 'boring', 'poor']

  const lowerText = text.toLowerCase()
  let positiveScore = 0
  let negativeScore = 0

  positiveWords.forEach(word => {
    const regex = new RegExp(`\\b${word}\\b`, 'g')
    positiveScore += (lowerText.match(regex) || []).length
  })

  negativeWords.forEach(word => {
    const regex = new RegExp(`\\b${word}\\b`, 'g')
    negativeScore += (lowerText.match(regex) || []).length
  })

  const total = positiveScore + negativeScore
  if (total === 0) {
    return {
      sentiment: 'positive' as const,
      confidence: 0.51,
      probabilities: { positive: 0.51, negative: 0.49 }
    }
  }

  const posProb = positiveScore / total
  const negProb = negativeScore / total

  return {
    sentiment: posProb > negProb ? 'positive' as const : 'negative' as const,
    confidence: Math.max(posProb, negProb),
    probabilities: { positive: posProb, negative: negProb }
  }
}

export async function POST(request: NextRequest) {
  try {
    const { text } = await request.json()

    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      return NextResponse.json(
        { error: 'Invalid input: text is required' },
        { status: 400 }
      )
    }

    const prediction = analyzeSentiment(text)

    return NextResponse.json(prediction)
  } catch (error) {
    console.error('Prediction error:', error)
    return NextResponse.json(
      { error: 'Failed to process prediction' },
      { status: 500 }
    )
  }
}
