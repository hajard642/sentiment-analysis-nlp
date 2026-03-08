'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { ArrowRight, Github, FileText } from 'lucide-react'

interface PredictionResult {
  sentiment: 'positive' | 'negative'
  confidence: number
  probabilities: {
    positive: number
    negative: number
  }
}

export default function Page() {
  const [text, setText] = useState('')
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handlePredict = async () => {
    if (!text.trim()) {
      setError('Please enter a review')
      return
    }

    setError('')
    setLoading(true)

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      })

      if (!response.ok) throw new Error('Prediction failed')
      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      setError('Failed to get prediction. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            Movie Review Sentiment Analysis
          </h1>
          <p className="text-xl text-slate-300">
            AI-powered sentiment classifier using DistilBERT
          </p>
          <p className="text-sm text-slate-400 mt-2">
            UPSaclay AI Master - Application Project
          </p>
        </div>

        {/* Main Card */}
        <Card className="mb-8 bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-2xl text-white">Analyze a Review</CardTitle>
            <CardDescription className="text-slate-300">
              Enter a movie review to predict its sentiment
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Input Section */}
            <div className="space-y-3">
              <label className="block text-sm font-medium text-slate-200">
                Movie Review
              </label>
              <Textarea
                placeholder="Enter a movie review... (e.g., 'This movie was absolutely fantastic! The acting was superb and the plot kept me engaged.')"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && e.ctrlKey) handlePredict()
                }}
                className="min-h-32 bg-slate-700 border-slate-600 text-white placeholder:text-slate-400 focus:border-blue-500"
              />
              <p className="text-xs text-slate-400">
                Ctrl+Enter to analyze
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 bg-red-900/20 border border-red-700 rounded text-red-200">
                {error}
              </div>
            )}

            {/* Prediction Result */}
            {prediction && (
              <div className="space-y-4 p-4 bg-slate-700/50 rounded-lg border border-slate-600">
                <div className="flex items-center justify-between">
                  <span className="text-slate-200 font-medium">Predicted Sentiment:</span>
                  <Badge 
                    className={`px-4 py-2 text-base ${
                      prediction.sentiment === 'positive' 
                        ? 'bg-green-600 hover:bg-green-700' 
                        : 'bg-red-600 hover:bg-red-700'
                    }`}
                  >
                    {prediction.sentiment.toUpperCase()}
                  </Badge>
                </div>

                <div className="space-y-2">
                  <p className="text-slate-200 font-medium">Confidence Score</p>
                  <div className="space-y-2">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-300">Positive</span>
                        <span className="text-green-400 font-semibold">
                          {(prediction.probabilities.positive * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-600 rounded-full h-2">
                        <div
                          className="bg-green-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${prediction.probabilities.positive * 100}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-300">Negative</span>
                        <span className="text-red-400 font-semibold">
                          {(prediction.probabilities.negative * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-600 rounded-full h-2">
                        <div
                          className="bg-red-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${prediction.probabilities.negative * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Predict Button */}
            <Button
              onClick={handlePredict}
              disabled={loading || !text.trim()}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white text-base py-6"
            >
              {loading ? (
                <>
                  <span className="animate-spin mr-2">⏳</span>
                  Analyzing...
                </>
              ) : (
                <>
                  Analyze Review
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Info Cards */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <FileText className="h-5 w-5 text-blue-400" />
                Project Details
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-2 text-sm">
              <p><strong>Model:</strong> DistilBERT (base, uncased)</p>
              <p><strong>Task:</strong> Binary Classification (Positive/Negative)</p>
              <p><strong>Training Data:</strong> 1,000 synthetic movie reviews</p>
              <p><strong>Train/Val/Test:</strong> 70% / 15% / 15%</p>
              <p><strong>Framework:</strong> PyTorch + Transformers</p>
              <p><strong>Loss Function:</strong> Cross-Entropy Loss</p>
              <p><strong>Optimizer:</strong> AdamW (lr=2e-5)</p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Github className="h-5 w-5 text-blue-400" />
                Resources
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-3 text-sm">
              <p className="font-medium text-white">Application Answers:</p>
              <div className="space-y-2">
                <a 
                  href="/docs/application-answers.md" 
                  className="text-blue-400 hover:text-blue-300 flex items-center gap-2"
                >
                  <ArrowRight className="h-4 w-4" /> Sections A-I
                </a>
                <a 
                  href="/docs/training-log.txt" 
                  className="text-blue-400 hover:text-blue-300 flex items-center gap-2"
                >
                  <ArrowRight className="h-4 w-4" /> Training Log
                </a>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Example Reviews */}
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Try Example Reviews</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid gap-2">
              {[
                {
                  text: "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
                  label: "Positive Example"
                },
                {
                  text: "Terrible film. The dialogue was awful, the characters were poorly developed, and I wasted my time.",
                  label: "Negative Example"
                }
              ].map((example, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setText(example.text)
                    setPrediction(null)
                  }}
                  className="p-4 bg-slate-700 hover:bg-slate-600 rounded-lg text-left transition-colors border border-slate-600"
                >
                  <p className="text-xs text-slate-400 mb-1">{example.label}</p>
                  <p className="text-slate-200">{example.text}</p>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
